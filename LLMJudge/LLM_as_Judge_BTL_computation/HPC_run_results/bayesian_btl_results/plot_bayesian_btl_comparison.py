"""
Generate a three-panel Bayesian comparison figure for human and LLM judges.

This script is intended to be run from the bayesian_btl_results directory.
It reads the fitted ArviZ/PyMC NetCDF files from the human, claude, gemini,
and gpt result subdirectories.

Outputs
-------
model_based_comparison.png
model_based_comparison.pdf
model_based_comparison_values.csv
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "model_based_comparison"

DIMENSIONS = ("Novelty", "Significance", "Testability")
EVALUATORS = ("Human", "Claude", "Gemini", "GPT")
LLM_EVALUATORS = ("Claude", "Gemini", "GPT")

FILES = {
    "Human": {
        "Novelty": Path("human/human_idata_Novelty.nc"),
        "Significance": Path("human/human_idata_Significance.nc"),
        "Testability": Path("human/human_idata_Testability.nc"),
    },
    "Claude": {
        "Novelty": Path("claude/claude_idata_Novelty.nc"),
        "Significance": Path("claude/claude_idata_Significance.nc"),
        "Testability": Path("claude/claude_idata_Testability.nc"),
    },
    "Gemini": {
        "Novelty": Path("gemini-3.5-flash-thinking/gemini-3.5-flash-thinking_idata_Novelty.nc"),
        "Significance": Path("gemini-3.5-flash-thinking/gemini-3.5-flash-thinking_idata_Significance.nc"),
        "Testability": Path("gemini-3.5-flash-thinking/gemini-3.5-flash-thinking_idata_Testability.nc"),
    },
    "GPT": {
        "Novelty": Path("gpt-5.5/gpt-5.5_idata_Novelty.nc"),
        "Significance": Path("gpt-5.5/gpt-5.5_idata_Significance.nc"),
        "Testability": Path("gpt-5.5/gpt-5.5_idata_Testability.nc"),
    },
}

COLORS = {
    "Human": "#999999",
    "Claude": "#015493",
    "Gemini": "#019092",
    "GPT": "#F4A99B",
}

MARKERS = {
    "Human": "o",
    "Claude": "s",
    "Gemini": "^",
    "GPT": "D",
}

ROPE_LOW = -0.1
ROPE_HIGH = 0.1
HDI_PROB = 0.95
N_DELTA_SAMPLES = 100_000
RANDOM_SEED = 42


def resolve_file(evaluator: str, dimension: str) -> Path:
    configured = SCRIPT_DIR / FILES[evaluator][dimension]
    if configured.exists():
        return configured

    evaluator_tokens = {
        "Human": ("human",),
        "Claude": ("claude",),
        "Gemini": ("gemini",),
        "GPT": ("gpt",),
    }[evaluator]

    candidates = []
    for path in SCRIPT_DIR.rglob("*.nc"):
        name = path.name.lower()
        evaluator_match = any(token in name for token in evaluator_tokens)
        dimension_match = dimension.lower() in name
        if evaluator_match and dimension_match:
            candidates.append(path)

    if len(candidates) == 1:
        print(f"Using {candidates[0].relative_to(SCRIPT_DIR)} for {evaluator} x {dimension}")
        return candidates[0]

    available = "\n".join(f"  - {path.relative_to(SCRIPT_DIR)}" for path in sorted(SCRIPT_DIR.rglob("*.nc")))
    if not available:
        available = "  No .nc files were found."

    raise FileNotFoundError(
        f"\nCannot uniquely locate the NetCDF file for {evaluator} x {dimension}.\n"
        f"Expected filename:\n  {FILES[evaluator][dimension]}\n\n"
        f"Available NetCDF files:\n{available}\n"
    )


def load_scalar_samples(file_path: Path, variable: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="btl_nc_") as tmp_dir:
        tmp_path = Path(tmp_dir) / file_path.name
        shutil.copy2(file_path, tmp_path)
        dataset = xr.open_dataset(tmp_path, group="posterior")
        dataset.load()
        dataset.close()

    available_variables = list(dataset.data_vars)
    if variable not in available_variables:
        raise KeyError(
            f"Variable {variable!r} was not found in {file_path.name}.\n"
            f"Available posterior variables: {available_variables}"
        )

    data_array = dataset[variable]
    extra_dimensions = [dim for dim in data_array.dims if dim not in ("chain", "draw")]
    if extra_dimensions:
        raise ValueError(
            f"{variable!r} in {file_path.name} was expected to be scalar, "
            f"but has extra dimensions: {extra_dimensions}"
        )

    samples = np.asarray(data_array.values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"{variable!r} in {file_path.name} contains no finite samples.")
    return samples


def calculate_hdi(samples: np.ndarray, probability: float = HDI_PROB) -> tuple[float, float]:
    sorted_samples = np.sort(np.asarray(samples, dtype=float))
    n = sorted_samples.size
    interval_width = int(np.floor(probability * n))
    if n < 2 or interval_width < 1:
        raise ValueError("At least two posterior samples are required to calculate an HDI.")

    widths = sorted_samples[interval_width:] - sorted_samples[: n - interval_width]
    min_index = int(np.argmin(widths))
    return float(sorted_samples[min_index]), float(sorted_samples[min_index + interval_width])


def summarize_samples(samples: np.ndarray) -> tuple[float, float, float]:
    lower, upper = calculate_hdi(samples)
    return float(np.mean(samples)), lower, upper


def baseline_tie_probability(tie_logit_samples: np.ndarray) -> np.ndarray:
    return np.exp(tie_logit_samples - np.logaddexp(np.log(2.0), tie_logit_samples))


def independent_posterior_difference(
    model_samples: np.ndarray,
    human_samples: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = N_DELTA_SAMPLES,
) -> np.ndarray:
    model_indices = rng.integers(0, model_samples.size, size=n_samples)
    human_indices = rng.integers(0, human_samples.size, size=n_samples)
    return model_samples[model_indices] - human_samples[human_indices]


def compute_results() -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED)
    posterior_samples: dict[tuple[str, str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []

    print("\nLoading posterior files")
    print("=" * 72)

    for evaluator in EVALUATORS:
        for dimension in DIMENSIONS:
            file_path = resolve_file(evaluator, dimension)
            print(f"{evaluator:8s} | {dimension:12s} | {file_path.relative_to(SCRIPT_DIR)}")

            mu_samples = load_scalar_samples(file_path, "mu_LLM")
            tie_logit_samples = load_scalar_samples(file_path, "tie_logit")
            tie_probability_samples = baseline_tie_probability(tie_logit_samples)

            posterior_samples[(evaluator, dimension, "mu_LLM")] = mu_samples
            posterior_samples[(evaluator, dimension, "tie_probability")] = tie_probability_samples

            mu_mean, mu_low, mu_high = summarize_samples(mu_samples)
            tie_mean, tie_low, tie_high = summarize_samples(tie_probability_samples)

            rows.append(
                {
                    "Panel": "source_effect",
                    "Evaluator": evaluator,
                    "Dimension": dimension,
                    "Parameter": "mu_LLM",
                    "Posterior_mean": mu_mean,
                    "HDI_low": mu_low,
                    "HDI_high": mu_high,
                    "P_above_zero": float(np.mean(mu_samples > 0)),
                    "P_in_ROPE": float(np.mean((mu_samples >= ROPE_LOW) & (mu_samples <= ROPE_HIGH))),
                }
            )
            rows.append(
                {
                    "Panel": "tie_probability",
                    "Evaluator": evaluator,
                    "Dimension": dimension,
                    "Parameter": "tie_probability",
                    "Posterior_mean": tie_mean,
                    "HDI_low": tie_low,
                    "HDI_high": tie_high,
                    "P_above_zero": np.nan,
                    "P_in_ROPE": np.nan,
                }
            )

    print("\nPosterior differences from human experts")
    print("=" * 72)

    delta_samples_for_plot: dict[tuple[str, str], np.ndarray] = {}
    for evaluator in LLM_EVALUATORS:
        for dimension in DIMENSIONS:
            model_mu = posterior_samples[(evaluator, dimension, "mu_LLM")]
            human_mu = posterior_samples[("Human", dimension, "mu_LLM")]
            delta_samples = independent_posterior_difference(model_mu, human_mu, rng)
            delta_samples_for_plot[(evaluator, dimension)] = delta_samples

            mean, low, high = summarize_samples(delta_samples)
            p_above_zero = float(np.mean(delta_samples > 0))
            p_in_rope = float(np.mean((delta_samples >= ROPE_LOW) & (delta_samples <= ROPE_HIGH)))

            print(
                f"{evaluator:7s} | {dimension:12s} | mean={mean: .3f}, "
                f"95% HDI=[{low: .3f}, {high: .3f}], "
                f"P(delta>0)={p_above_zero:.3f}, P(ROPE)={p_in_rope:.3f}"
            )

            rows.append(
                {
                    "Panel": "difference_from_human",
                    "Evaluator": evaluator,
                    "Dimension": dimension,
                    "Parameter": "delta_mu_vs_human",
                    "Posterior_mean": mean,
                    "HDI_low": low,
                    "HDI_high": high,
                    "P_above_zero": p_above_zero,
                    "P_in_ROPE": p_in_rope,
                }
            )

    return pd.DataFrame(rows), delta_samples_for_plot


def ordered_subset(results: pd.DataFrame, panel: str) -> pd.DataFrame:
    subset = results[results["Panel"] == panel].copy()
    subset["Dimension"] = pd.Categorical(subset["Dimension"], categories=DIMENSIONS, ordered=True)
    subset["Evaluator"] = pd.Categorical(subset["Evaluator"], categories=EVALUATORS, ordered=True)
    return subset.sort_values(["Dimension", "Evaluator"])


def add_reference_elements(ax: plt.Axes, show_rope: bool) -> None:
    if show_rope:
        ax.axvspan(ROPE_LOW, ROPE_HIGH, color="0.70", alpha=0.15, linewidth=0, zorder=0)
        ax.axvline(ROPE_LOW, color="0.60", linestyle=(0, (1.5, 2.5)), linewidth=0.7, zorder=1)
        ax.axvline(ROPE_HIGH, color="0.60", linestyle=(0, (1.5, 2.5)), linewidth=0.7, zorder=1)
    ax.axvline(0, color="0.25", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)


def draw_grouped_intervals(
    ax: plt.Axes,
    data: pd.DataFrame,
    evaluators: tuple[str, ...],
    offsets: dict[str, float],
    xlim: tuple[float, float],
    xticks: list[float],
    xlabel: str,
    title: str,
    show_ylabels: bool,
    show_rope: bool,
    percent_axis: bool = False,
) -> None:
    base_y = {"Novelty": 2.0, "Significance": 1.0, "Testability": 0.0}
    ax.set_axisbelow(True)

    for y in base_y.values():
        ax.axhline(y, color="0.92", linewidth=0.7, zorder=0)

    if show_rope:
        add_reference_elements(ax, show_rope=True)

    for evaluator in evaluators:
        evaluator_data = data[data["Evaluator"].astype(str) == evaluator]
        for _, row in evaluator_data.iterrows():
            dimension = str(row["Dimension"])
            y_value = base_y[dimension] + offsets[evaluator]
            mean = float(row["Posterior_mean"])
            low = float(row["HDI_low"])
            high = float(row["HDI_high"])
            asymmetric_error = np.array([[mean - low], [high - mean]])

            ax.errorbar(
                mean,
                y_value,
                xerr=asymmetric_error,
                fmt=MARKERS[evaluator],
                markersize=5.4,
                markerfacecolor=COLORS[evaluator],
                markeredgecolor="white",
                markeredgewidth=0.7,
                ecolor=COLORS[evaluator],
                elinewidth=1.6,
                capsize=2.6,
                capthick=1.0,
                zorder=3,
            )

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.48, 2.48)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="semibold", pad=8)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_yticks([2.0, 1.0, 0.0])

    if show_ylabels:
        ax.set_yticklabels(DIMENSIONS)
        ax.tick_params(axis="y", length=0, pad=6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
        ax.spines["left"].set_visible(False)

    if percent_axis:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.tick_params(axis="x", length=3, width=0.8, color="0.25", pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    if show_ylabels:
        ax.spines["left"].set_linewidth(0.8)


def make_figure(results: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    source_data = ordered_subset(results, "source_effect")
    difference_data = ordered_subset(results, "difference_from_human")
    tie_data = ordered_subset(results, "tie_probability")

    fig = plt.figure(figsize=(8.4, 2.85))
    grid = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.18, 1.18, 0.92], wspace=0.33)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])]

    source_offsets = {"Human": 0.24, "Claude": 0.08, "Gemini": -0.08, "GPT": -0.24}
    difference_offsets = {"Claude": 0.16, "Gemini": 0.00, "GPT": -0.16}

    draw_grouped_intervals(
        ax=axes[0],
        data=source_data,
        evaluators=EVALUATORS,
        offsets=source_offsets,
        xlim=(-1.4, 5.5),
        xticks=[-1, 0, 1, 2, 3, 4, 5],
        xlabel=r"$\mu_{\mathrm{LLM}}$",
        title="Overall source effect",
        show_ylabels=True,
        show_rope=True,
    )

    draw_grouped_intervals(
        ax=axes[1],
        data=difference_data,
        evaluators=LLM_EVALUATORS,
        offsets=difference_offsets,
        xlim=(-1.4, 5.5),
        xticks=[-1, 0, 1, 2, 3, 4, 5],
        xlabel=r"$\Delta\mu$ relative to human experts",
        title="Difference from human experts",
        show_ylabels=False,
        show_rope=True,
    )

    draw_grouped_intervals(
        ax=axes[2],
        data=tie_data,
        evaluators=EVALUATORS,
        offsets=source_offsets,
        xlim=(0.0, 0.30),
        xticks=[0.0, 0.1, 0.2, 0.3],
        xlabel=r"$P(\mathrm{Tie}\mid\Delta=0)$",
        title="Baseline tie probability",
        show_ylabels=False,
        show_rope=False,
        percent_axis=True,
    )

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(-0.15, 1.10, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="left")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[evaluator],
            linestyle="none",
            markerfacecolor=COLORS[evaluator],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6,
            label=evaluator,
        )
        for evaluator in EVALUATORS
    ]

    fig.legend(
        handles=legend_handles,
        labels=EVALUATORS,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.04),
        ncol=4,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.3,
        fontsize=8,
    )

    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.24, top=0.78)

    png_path = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = SCRIPT_DIR / f"{OUTPUT_STEM}.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved figure files")
    print("=" * 72)
    print(png_path)
    print(pdf_path)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    results, _ = compute_results()

    csv_path = SCRIPT_DIR / f"{OUTPUT_STEM}_values.csv"
    results.to_csv(csv_path, index=False, float_format="%.10g")
    make_figure(results)
    print(csv_path)


if __name__ == "__main__":
    main()
