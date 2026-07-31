"""
Generate a three-panel Bayesian summary figure for LLM-vs-human hypothesis ratings.

Outputs
-------
Bayesian_summary.png
Bayesian_summary.pdf
Bayesian_summary_values.csv
"""

from __future__ import annotations

from pathlib import Path
import importlib.machinery
import sys
import types
import warnings

import numpy as np


def patch_optional_dependency_issues() -> None:
    """Keep older ArviZ installations usable with newer SciPy or broken numba."""
    warnings.filterwarnings(
        "ignore",
        message=r"Pandas requires version .*",
        category=UserWarning,
    )

    try:
        import scipy.signal as scipy_signal
        from scipy.signal import windows as scipy_windows

        if not hasattr(scipy_signal, "gaussian") and hasattr(scipy_windows, "gaussian"):
            scipy_signal.gaussian = scipy_windows.gaussian
    except Exception:
        pass

    try:
        import numba  # noqa: F401
    except Exception:
        dummy_numba = types.ModuleType("numba")
        dummy_numba.__spec__ = importlib.machinery.ModuleSpec("numba", loader=None)

        def identity_decorator(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def decorate(func):
                return func

            return decorate

        dummy_numba.jit = identity_decorator
        dummy_numba.vectorize = identity_decorator
        sys.modules["numba"] = dummy_numba


patch_optional_dependency_issues()

import arviz as az  # noqa: E402
import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "Bayesian_summary"

DIMENSIONS = ("Novelty", "Significance", "Testability")
PARAMETERS = ("mu_LLM", "sigma_LLM", "tie_logit")
COLORS = {
    "Novelty": "#1f77b4",
    "Significance": "#ff7f0e",
    "Testability": "#2ca02c",
}


def resolve_netcdf_file(directory: Path, dimension: str) -> Path:
    """
    Find the NetCDF file for one dimension.

    Exact names such as idata_Novelty.nc are preferred. If an exact file is absent,
    names containing parentheses or numbers, such as idata_Novelty (1).nc, are allowed.
    """
    nc_files = sorted(directory.glob("*.nc"))
    exact_name = f"idata_{dimension}.nc".lower()
    for path in nc_files:
        if path.name.lower() == exact_name:
            return path

    candidates = [
        path
        for path in nc_files
        if dimension.lower() in path.name.lower() and "idata" in path.name.lower()
    ]
    if not candidates:
        available = ", ".join(path.name for path in nc_files) or "no .nc files found"
        raise FileNotFoundError(
            f"Cannot find a NetCDF file for {dimension!r} in {directory}.\n"
            f"Available NetCDF files: {available}"
        )

    candidates = sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )
    chosen = candidates[0]
    print(
        f"Using {chosen.name} for {dimension}; "
        f"other matching files: {', '.join(path.name for path in candidates[1:]) or 'none'}"
    )
    return chosen


def get_scalar_posterior_samples(idata: az.InferenceData, variable: str, file_path: Path) -> np.ndarray:
    """Return flattened chain/draw posterior samples for one scalar variable."""
    if not hasattr(idata, "posterior"):
        raise ValueError(f"{file_path.name} does not contain a posterior group.")

    posterior_vars = list(idata.posterior.data_vars)
    if variable not in posterior_vars:
        available = ", ".join(posterior_vars) or "none"
        raise KeyError(
            f"Variable {variable!r} was not found in posterior for {file_path.name}.\n"
            f"Available posterior variables: {available}"
        )

    da = idata.posterior[variable]
    extra_dims = [dim for dim in da.dims if dim not in ("chain", "draw")]
    if extra_dims:
        raise ValueError(
            f"Expected {variable!r} in {file_path.name} to be scalar over chain/draw, "
            f"but found extra dimensions: {extra_dims}."
        )

    samples = np.asarray(da.values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"Variable {variable!r} in {file_path.name} has no finite samples.")
    return samples


def hdi_95(samples: np.ndarray) -> tuple[float, float]:
    """Compute the 95% highest density interval with ArviZ."""
    hdi = np.asarray(az.hdi(samples, hdi_prob=0.95), dtype=float).reshape(-1)
    if hdi.size != 2:
        raise ValueError(f"Expected a two-value HDI, but got shape {hdi.shape}.")
    return float(hdi[0]), float(hdi[1])


def tie_probability_from_logit(tie_logit_samples: np.ndarray) -> np.ndarray:
    """
    Convert Davidson tie logits to model-implied tie probabilities at delta = 0.

    P_tie = exp(tie_logit) / (2 + exp(tie_logit)).
    The logaddexp form is numerically stable and applies the transformation to
    every posterior sample before any summaries are computed.
    """
    return np.exp(tie_logit_samples - np.logaddexp(np.log(2.0), tie_logit_samples))


def summarize_samples(samples: np.ndarray) -> tuple[float, float, float]:
    """Return posterior mean and 95% HDI."""
    lower, upper = hdi_95(samples)
    return float(np.mean(samples)), lower, upper


def compute_results() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Read all NetCDF files, compute summaries, and print diagnostics."""
    plot_rows = []
    diagnostics: dict[str, dict[str, float]] = {}

    print("\nBayesian posterior summaries used for plotting")
    print("=" * 60)

    for dimension in DIMENSIONS:
        file_path = resolve_netcdf_file(SCRIPT_DIR, dimension)
        idata = az.from_netcdf(file_path)

        for variable in PARAMETERS:
            get_scalar_posterior_samples(idata, variable, file_path)

        mu_samples = get_scalar_posterior_samples(idata, "mu_LLM", file_path)
        sigma_samples = get_scalar_posterior_samples(idata, "sigma_LLM", file_path)
        tie_logit_samples = get_scalar_posterior_samples(idata, "tie_logit", file_path)
        tie_probability_samples = tie_probability_from_logit(tie_logit_samples)

        mu_mean, mu_low, mu_high = summarize_samples(mu_samples)
        sigma_mean, sigma_low, sigma_high = summarize_samples(sigma_samples)
        tie_prob_mean, tie_prob_low, tie_prob_high = summarize_samples(tie_probability_samples)
        tie_logit_mean = float(np.mean(tie_logit_samples))

        p_mu_gt_0 = float(np.mean(mu_samples > 0.0))
        p_mu_gt_01 = float(np.mean(mu_samples > 0.1))
        p_mu_rope = float(np.mean((mu_samples >= -0.1) & (mu_samples <= 0.1)))

        print(f"\nDimension: {dimension}")
        print(f"  Source file: {file_path.name}")
        print(f"  mu_LLM mean and 95% HDI: {mu_mean:.4f} [{mu_low:.4f}, {mu_high:.4f}]")
        print(
            f"  sigma_LLM mean and 95% HDI: "
            f"{sigma_mean:.4f} [{sigma_low:.4f}, {sigma_high:.4f}]"
        )
        print(f"  tie_logit mean: {tie_logit_mean:.4f}")
        print(
            f"  transformed tie probability mean and 95% HDI: "
            f"{tie_prob_mean:.4f} [{tie_prob_low:.4f}, {tie_prob_high:.4f}]"
        )
        print(f"  P(mu_LLM > 0): {p_mu_gt_0:.4f}")
        print(f"  P(mu_LLM > 0.1): {p_mu_gt_01:.4f}")
        print(f"  P(mu_LLM within [-0.1, 0.1]): {p_mu_rope:.4f}")

        plot_rows.extend(
            [
                {
                    "Dimension": dimension,
                    "Parameter": "mu_LLM",
                    "Posterior_mean": mu_mean,
                    "HDI_2.5": mu_low,
                    "HDI_97.5": mu_high,
                },
                {
                    "Dimension": dimension,
                    "Parameter": "sigma_LLM",
                    "Posterior_mean": sigma_mean,
                    "HDI_2.5": sigma_low,
                    "HDI_97.5": sigma_high,
                },
                {
                    "Dimension": dimension,
                    "Parameter": "tie_probability",
                    "Posterior_mean": tie_prob_mean,
                    "HDI_2.5": tie_prob_low,
                    "HDI_97.5": tie_prob_high,
                },
            ]
        )

        diagnostics[dimension] = {
            "tie_logit_mean": tie_logit_mean,
            "p_mu_gt_0": p_mu_gt_0,
            "p_mu_gt_0.1": p_mu_gt_01,
            "p_mu_within_rope": p_mu_rope,
        }

    return pd.DataFrame(plot_rows), diagnostics


def subset_for_parameter(results: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Return rows for one parameter in the requested dimension order."""
    subset = results[results["Parameter"] == parameter].copy()
    subset["Dimension"] = pd.Categorical(subset["Dimension"], categories=DIMENSIONS, ordered=True)
    return subset.sort_values("Dimension")


def draw_horizontal_interval_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    xlabel: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    percent_axis: bool = False,
    reference_zero: bool = False,
    rope: tuple[float, float] | None = None,
) -> None:
    """Draw one restrained horizontal point-interval panel."""
    y_positions = np.arange(len(DIMENSIONS))[::-1]
    y_lookup = dict(zip(DIMENSIONS, y_positions))

    if rope is not None:
        ax.axvspan(rope[0], rope[1], color="0.88", alpha=0.8, lw=0, zorder=0)

    if reference_zero:
        ax.axvline(0, color="0.35", linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)

    for _, row in data.iterrows():
        dimension = str(row["Dimension"])
        y = y_lookup[dimension]
        color = COLORS[dimension]
        lower = float(row["HDI_2.5"])
        upper = float(row["HDI_97.5"])
        mean = float(row["Posterior_mean"])

        ax.hlines(y, lower, upper, color=color, linewidth=2.8, zorder=2)
        ax.scatter(
            mean,
            y,
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(DIMENSIONS)
    ax.set_ylim(-0.6, len(DIMENSIONS) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=8)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if percent_axis:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.tick_params(axis="both", length=3, width=0.7, color="0.25")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def padded_xlim(values_low: np.ndarray, values_high: np.ndarray, include: tuple[float, ...]) -> tuple[float, float]:
    """Create a compact x-axis range with a small margin."""
    xmin = float(np.min(np.r_[values_low, include]))
    xmax = float(np.max(np.r_[values_high, include]))
    span = xmax - xmin
    pad = 0.08 * span if span > 0 else 0.1
    return xmin - pad, xmax + pad


def make_figure(results: pd.DataFrame) -> None:
    """Create and save the three-panel figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    mu_data = subset_for_parameter(results, "mu_LLM")
    sigma_data = subset_for_parameter(results, "sigma_LLM")
    tie_data = subset_for_parameter(results, "tie_probability")

    mu_xlim = padded_xlim(
        mu_data["HDI_2.5"].to_numpy(),
        mu_data["HDI_97.5"].to_numpy(),
        include=(-0.1, 0.0, 0.1),
    )
    sigma_xlim = padded_xlim(
        sigma_data["HDI_2.5"].to_numpy(),
        sigma_data["HDI_97.5"].to_numpy(),
        include=(0.0,),
    )
    sigma_xlim = (0.0, sigma_xlim[1])
    tie_xlim = padded_xlim(
        tie_data["HDI_2.5"].to_numpy(),
        tie_data["HDI_97.5"].to_numpy(),
        include=(0.0,),
    )
    tie_xlim = (0.0, min(1.0, tie_xlim[1]))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7.2, 2.65), constrained_layout=False)

    draw_horizontal_interval_panel(
        axes[0],
        mu_data,
        xlabel="Population-level LLM source effect",
        title="Population-level source effect",
        xlim=mu_xlim,
        reference_zero=True,
        rope=(-0.1, 0.1),
    )
    draw_horizontal_interval_panel(
        axes[1],
        sigma_data,
        xlabel="Inter-expert heterogeneity",
        title="Inter-expert heterogeneity",
        xlim=sigma_xlim,
    )
    draw_horizontal_interval_panel(
        axes[2],
        tie_data,
        xlabel="Model-implied tie probability",
        title="Tie probability",
        xlim=tie_xlim,
        percent_axis=True,
    )

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.18,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.tight_layout(w_pad=1.6)
    fig.savefig(SCRIPT_DIR / f"{OUTPUT_STEM}.png", dpi=600, bbox_inches="tight")
    fig.savefig(SCRIPT_DIR / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results, _ = compute_results()
    csv_path = SCRIPT_DIR / f"{OUTPUT_STEM}_values.csv"
    results.to_csv(csv_path, index=False, float_format="%.10g")
    make_figure(results)

    print("\nSaved outputs")
    print("=" * 60)
    print(SCRIPT_DIR / f"{OUTPUT_STEM}.png")
    print(SCRIPT_DIR / f"{OUTPUT_STEM}.pdf")
    print(csv_path)


if __name__ == "__main__":
    main()
