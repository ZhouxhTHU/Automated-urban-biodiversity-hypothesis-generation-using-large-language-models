"""
Generate a three-panel Bayesian summary figure and an Excel report for
LLM-vs-human hypothesis ratings.

Expected folder structure
-------------------------
project/
├─ ResultsFinal/
│  ├─ idata_Novelty.nc
│  ├─ idata_Significance.nc
│  └─ idata_Testability.nc
└─ figure/
   └─ Bayesian_summary_with_excel.py

Outputs
-------
Bayesian_summary.png
Bayesian_summary.pdf
Bayesian_summary_values.csv
Bayesian_summary.xlsx

The Excel workbook contains:
1. Posterior_summary   — source effects, heterogeneity, tie probabilities and ROPE probabilities
2. Model_diagnostics   — divergences, maximum R-hat and minimum ESS by dimension
3. Diagnostic_details  — R-hat and ESS for every posterior parameter
4. Plot_values         — exact values used in the three figure panels
5. Manuscript_text     — ready-to-copy Results paragraphs generated from the current files
6. README              — definitions and calculation notes
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
import xarray as xr  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "ResultsFinal"

OUTPUT_STEM = "Bayesian_summary"
EXCEL_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}.xlsx"
CSV_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_values.csv"

DIMENSIONS = ("Novelty", "Significance", "Testability")
PARAMETERS = ("mu_LLM", "sigma_LLM", "tie_logit")
HDI_PROB = 0.95
ROPE_LOW = -0.1
ROPE_HIGH = 0.1

# Optional explicit file names. Leave a value as None to use automatic resolution.
# Explicit names are recommended if ResultsFinal contains several versions.
NETCDF_FILES: dict[str, str | None] = {
    "Novelty": None,
    "Significance": None,  # Example: "idata_Significance(6).nc"
    "Testability": None,   # Example: "idata_Testability(7).nc"
}

COLORS = {
    "Novelty": "#0072B2",      # blue
    "Significance": "#E69F00", # orange
    "Testability": "#CC79A7",  # reddish purple
}

def resolve_netcdf_file(directory: Path, dimension: str) -> Path:
    """Resolve one NetCDF file without silently selecting among multiple versions."""
    explicit_name = NETCDF_FILES.get(dimension)
    if explicit_name:
        path = directory / explicit_name
        if not path.exists():
            raise FileNotFoundError(
                f"Explicit NetCDF file for {dimension!r} does not exist: {path}"
            )
        return path

    exact_path = directory / f"idata_{dimension}.nc"
    if exact_path.exists():
        return exact_path

    candidates = sorted(
        path
        for path in directory.glob("*.nc")
        if dimension.lower() in path.name.lower() and "idata" in path.name.lower()
    )

    if not candidates:
        available = ", ".join(path.name for path in directory.glob("*.nc")) or "none"
        raise FileNotFoundError(
            f"Cannot find a NetCDF file for {dimension!r} in {directory}.\n"
            f"Available NetCDF files: {available}"
        )

    if len(candidates) > 1:
        candidate_names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Multiple NetCDF files match {dimension!r}: {candidate_names}.\n"
            f"Set NETCDF_FILES[{dimension!r}] near the top of the script to the exact file name."
        )

    return candidates[0]


def _decode_coordinate(values: np.ndarray) -> list[str]:
    """Decode byte-string coordinates read through h5py."""
    decoded = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _load_netcdf_with_h5py(file_path: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Fallback reader for environments where xarray lacks netCDF4/h5netcdf.

    The project files use chain, draw and expert dimensions, which are reconstructed
    here. The normal az.from_netcdf route remains preferred.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "ArviZ could not read the NetCDF file and h5py is unavailable for fallback reading."
        ) from exc

    with h5py.File(file_path, "r") as handle:
        if "posterior" not in handle or "sample_stats" not in handle:
            raise ValueError(
                f"{file_path.name} must contain posterior and sample_stats groups."
            )

        posterior_group = handle["posterior"]
        chain = np.asarray(posterior_group["chain"])
        draw = np.asarray(posterior_group["draw"])
        posterior_coords: dict[str, object] = {"chain": chain, "draw": draw}

        if "expert" in posterior_group:
            posterior_coords["expert"] = _decode_coordinate(
                np.asarray(posterior_group["expert"])
            )

        posterior_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for name, dataset in posterior_group.items():
            if name in {"chain", "draw", "expert"}:
                continue
            values = np.asarray(dataset)
            if values.ndim == 2:
                dims = ("chain", "draw")
            elif (
                values.ndim == 3
                and "expert" in posterior_coords
                and values.shape[2] == len(posterior_coords["expert"])
            ):
                dims = ("chain", "draw", "expert")
            else:
                extra_dims = tuple(
                    f"{name}_dim_{index}" for index in range(values.ndim - 2)
                )
                dims = ("chain", "draw", *extra_dims)
            posterior_vars[name] = (dims, values)

        posterior = xr.Dataset(posterior_vars, coords=posterior_coords)

        stats_group = handle["sample_stats"]
        stats_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for name, dataset in stats_group.items():
            if name in {"chain", "draw"}:
                continue
            values = np.asarray(dataset)
            if values.ndim == 2:
                dims = ("chain", "draw")
            else:
                extra_dims = tuple(
                    f"{name}_dim_{index}" for index in range(values.ndim - 2)
                )
                dims = ("chain", "draw", *extra_dims)
            stats_vars[name] = (dims, values)

        sample_stats = xr.Dataset(
            stats_vars,
            coords={"chain": chain, "draw": draw},
        )

    return posterior, sample_stats


def load_model_data(file_path: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Load posterior and sample-statistics datasets from an ArviZ NetCDF file."""
    try:
        idata = az.from_netcdf(file_path)
        if not hasattr(idata, "posterior") or not hasattr(idata, "sample_stats"):
            raise ValueError("The NetCDF file lacks posterior or sample_stats data.")
        return idata.posterior, idata.sample_stats
    except Exception as primary_error:
        print(
            f"Standard ArviZ loading failed for {file_path.name}; "
            "using the h5py fallback reader."
        )
        try:
            return _load_netcdf_with_h5py(file_path)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not load {file_path}.\n"
                f"ArviZ error: {primary_error}\n"
                f"Fallback error: {fallback_error}"
            ) from fallback_error


def get_scalar_posterior_samples(
    posterior: xr.Dataset,
    variable: str,
    file_path: Path,
) -> np.ndarray:
    """Return flattened chain/draw posterior samples for one scalar variable."""
    if variable not in posterior.data_vars:
        available = ", ".join(posterior.data_vars) or "none"
        raise KeyError(
            f"Variable {variable!r} was not found in posterior for {file_path.name}.\n"
            f"Available posterior variables: {available}"
        )

    data_array = posterior[variable]
    extra_dims = [dim for dim in data_array.dims if dim not in ("chain", "draw")]
    if extra_dims:
        raise ValueError(
            f"Expected {variable!r} in {file_path.name} to be scalar over chain/draw, "
            f"but found extra dimensions: {extra_dims}."
        )

    samples = np.asarray(data_array.values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"Variable {variable!r} in {file_path.name} has no finite samples.")
    return samples


def hdi_interval(samples: np.ndarray, probability: float = HDI_PROB) -> tuple[float, float]:
    """Compute an HDI while supporting both older and newer ArviZ argument names."""
    try:
        interval = az.hdi(samples, hdi_prob=probability)
    except TypeError:
        interval = az.hdi(samples, prob=probability)

    values = np.asarray(interval, dtype=float).reshape(-1)
    if values.size != 2:
        raise ValueError(f"Expected two HDI endpoints, but got shape {values.shape}.")
    return float(values[0]), float(values[1])


def tie_probability_from_logit(tie_logit_samples: np.ndarray) -> np.ndarray:
    """
    Convert Davidson tie logits to model-implied tie probabilities at delta = 0.

    P(Tie | delta=0) = exp(tie_logit) / (2 + exp(tie_logit)).
    The transformation is applied to every posterior sample before summarization.
    """
    return np.exp(
        tie_logit_samples - np.logaddexp(np.log(2.0), tie_logit_samples)
    )


def summarize_samples(samples: np.ndarray) -> dict[str, float]:
    """Return posterior mean, standard deviation and 95% HDI."""
    lower, upper = hdi_interval(samples)
    return {
        "Posterior_mean": float(np.mean(samples)),
        "Posterior_SD": float(np.std(samples, ddof=1)),
        "HDI_95_lower": lower,
        "HDI_95_upper": upper,
    }


def flatten_diagnostic_dataset(dataset: xr.Dataset, column_name: str) -> pd.DataFrame:
    """Convert scalar and indexed ArviZ diagnostic results into a long table."""
    rows: list[dict[str, object]] = []

    for variable in dataset.data_vars:
        data_array = dataset[variable]
        values = np.asarray(data_array.values, dtype=float)

        if values.ndim == 0:
            rows.append(
                {
                    "Parameter": variable,
                    column_name: float(values),
                }
            )
            continue

        for index_tuple in np.ndindex(values.shape):
            coordinate_labels = []
            for dim_name, coordinate_index in zip(data_array.dims, index_tuple):
                if dim_name in data_array.coords:
                    coordinate_value = data_array.coords[dim_name].values[coordinate_index]
                    if isinstance(coordinate_value, (bytes, np.bytes_)):
                        coordinate_value = coordinate_value.decode("utf-8")
                    coordinate_labels.append(str(coordinate_value))
                else:
                    coordinate_labels.append(str(coordinate_index))

            parameter_name = f"{variable}[{','.join(coordinate_labels)}]"
            rows.append(
                {
                    "Parameter": parameter_name,
                    column_name: float(values[index_tuple]),
                }
            )

    return pd.DataFrame(rows)


def compute_diagnostics(
    posterior: xr.Dataset,
    sample_stats: xr.Dataset,
    dimension: str,
    file_path: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute convergence diagnostics for all posterior variables."""
    rhat_dataset = az.rhat(posterior)
    ess_bulk_dataset = az.ess(posterior, method="bulk")
    ess_tail_dataset = az.ess(posterior, method="tail")

    rhat_table = flatten_diagnostic_dataset(rhat_dataset, "R_hat")
    bulk_table = flatten_diagnostic_dataset(ess_bulk_dataset, "ESS_bulk")
    tail_table = flatten_diagnostic_dataset(ess_tail_dataset, "ESS_tail")

    detail = rhat_table.merge(bulk_table, on="Parameter", how="outer").merge(
        tail_table, on="Parameter", how="outer"
    )
    detail.insert(0, "Dimension", dimension)
    detail.insert(1, "Source_file", file_path.name)

    valid_rhat = detail.dropna(subset=["R_hat"])
    valid_bulk = detail.dropna(subset=["ESS_bulk"])
    valid_tail = detail.dropna(subset=["ESS_tail"])

    max_rhat_row = valid_rhat.loc[valid_rhat["R_hat"].idxmax()]
    min_bulk_row = valid_bulk.loc[valid_bulk["ESS_bulk"].idxmin()]
    min_tail_row = valid_tail.loc[valid_tail["ESS_tail"].idxmin()]

    if "diverging" in sample_stats.data_vars:
        divergent_transitions = int(np.asarray(sample_stats["diverging"]).sum())
    elif "divergences" in sample_stats.data_vars:
        divergent_transitions = int(np.asarray(sample_stats["divergences"]).sum())
    else:
        divergent_transitions = 0

    reached_max_treedepth = None
    if "reached_max_treedepth" in sample_stats.data_vars:
        reached_max_treedepth = int(
            np.asarray(sample_stats["reached_max_treedepth"]).sum()
        )

    chains = int(posterior.sizes.get("chain", 1))
    draws_per_chain = int(posterior.sizes.get("draw", 1))

    summary = {
        "Dimension": dimension,
        "Source_file": file_path.name,
        "Chains": chains,
        "Draws_per_chain": draws_per_chain,
        "Total_posterior_draws": chains * draws_per_chain,
        "Divergent_transitions": divergent_transitions,
        "Reached_max_treedepth": reached_max_treedepth,
        "Maximum_R_hat": float(max_rhat_row["R_hat"]),
        "Maximum_R_hat_parameter": str(max_rhat_row["Parameter"]),
        "Minimum_ESS_bulk": float(min_bulk_row["ESS_bulk"]),
        "Minimum_ESS_bulk_parameter": str(min_bulk_row["Parameter"]),
        "Minimum_ESS_tail": float(min_tail_row["ESS_tail"]),
        "Minimum_ESS_tail_parameter": str(min_tail_row["Parameter"]),
    }

    return summary, detail


def compute_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Read all NetCDF files and compute plot values, summaries and diagnostics."""
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find ResultsFinal directory: {RESULTS_DIR}\n"
            f"Current script directory is: {SCRIPT_DIR}"
        )

    plot_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    diagnostic_detail_tables: list[pd.DataFrame] = []

    print("\nBayesian posterior summaries and diagnostics")
    print("=" * 72)

    for dimension in DIMENSIONS:
        file_path = resolve_netcdf_file(RESULTS_DIR, dimension)
        posterior, sample_stats = load_model_data(file_path)

        for variable in PARAMETERS:
            get_scalar_posterior_samples(posterior, variable, file_path)

        mu_samples = get_scalar_posterior_samples(posterior, "mu_LLM", file_path)
        sigma_samples = get_scalar_posterior_samples(posterior, "sigma_LLM", file_path)
        tie_logit_samples = get_scalar_posterior_samples(
            posterior, "tie_logit", file_path
        )
        tie_probability_samples = tie_probability_from_logit(tie_logit_samples)

        mu_summary = summarize_samples(mu_samples)
        sigma_summary = summarize_samples(sigma_samples)
        tie_logit_summary = summarize_samples(tie_logit_samples)
        tie_probability_summary = summarize_samples(tie_probability_samples)

        p_mu_gt_0 = float(np.mean(mu_samples > 0.0))
        p_mu_gt_01 = float(np.mean(mu_samples > 0.1))
        p_mu_lt_minus_01 = float(np.mean(mu_samples < -0.1))
        p_mu_rope = float(
            np.mean((mu_samples >= ROPE_LOW) & (mu_samples <= ROPE_HIGH))
        )

        posterior_rows.append(
            {
                "Dimension": dimension,
                "Source_file": file_path.name,
                "Posterior_draws": int(mu_samples.size),
                "mu_LLM_mean": mu_summary["Posterior_mean"],
                "mu_LLM_SD": mu_summary["Posterior_SD"],
                "mu_LLM_HDI95_lower": mu_summary["HDI_95_lower"],
                "mu_LLM_HDI95_upper": mu_summary["HDI_95_upper"],
                "P_mu_gt_0": p_mu_gt_0,
                "P_mu_gt_0.1": p_mu_gt_01,
                "P_mu_lt_minus_0.1": p_mu_lt_minus_01,
                "P_mu_within_ROPE": p_mu_rope,
                "sigma_LLM_mean": sigma_summary["Posterior_mean"],
                "sigma_LLM_SD": sigma_summary["Posterior_SD"],
                "sigma_LLM_HDI95_lower": sigma_summary["HDI_95_lower"],
                "sigma_LLM_HDI95_upper": sigma_summary["HDI_95_upper"],
                "tie_logit_mean": tie_logit_summary["Posterior_mean"],
                "tie_probability_mean": tie_probability_summary["Posterior_mean"],
                "tie_probability_HDI95_lower": tie_probability_summary["HDI_95_lower"],
                "tie_probability_HDI95_upper": tie_probability_summary["HDI_95_upper"],
            }
        )

        for parameter, summary in (
            ("mu_LLM", mu_summary),
            ("sigma_LLM", sigma_summary),
            ("tie_probability", tie_probability_summary),
        ):
            plot_rows.append(
                {
                    "Dimension": dimension,
                    "Parameter": parameter,
                    **summary,
                }
            )

        diagnostic_summary, diagnostic_detail = compute_diagnostics(
            posterior,
            sample_stats,
            dimension,
            file_path,
        )
        diagnostic_rows.append(diagnostic_summary)
        diagnostic_detail_tables.append(diagnostic_detail)

        print(f"\nDimension: {dimension}")
        print(f"  Source file: {file_path.name}")
        print(
            "  mu_LLM mean and 95% HDI: "
            f"{mu_summary['Posterior_mean']:.4f} "
            f"[{mu_summary['HDI_95_lower']:.4f}, {mu_summary['HDI_95_upper']:.4f}]"
        )
        print(
            "  sigma_LLM mean and 95% HDI: "
            f"{sigma_summary['Posterior_mean']:.4f} "
            f"[{sigma_summary['HDI_95_lower']:.4f}, {sigma_summary['HDI_95_upper']:.4f}]"
        )
        print(
            "  tie probability mean and 95% HDI: "
            f"{tie_probability_summary['Posterior_mean']:.4f} "
            f"[{tie_probability_summary['HDI_95_lower']:.4f}, "
            f"{tie_probability_summary['HDI_95_upper']:.4f}]"
        )
        print(f"  P(mu_LLM > 0): {p_mu_gt_0:.4f}")
        print(f"  P(mu_LLM > 0.1): {p_mu_gt_01:.4f}")
        print(f"  P(mu_LLM in ROPE): {p_mu_rope:.4f}")
        print(
            "  divergences / max R-hat / min bulk ESS / min tail ESS: "
            f"{diagnostic_summary['Divergent_transitions']} / "
            f"{diagnostic_summary['Maximum_R_hat']:.4f} / "
            f"{diagnostic_summary['Minimum_ESS_bulk']:.1f} / "
            f"{diagnostic_summary['Minimum_ESS_tail']:.1f}"
        )

    plot_results = pd.DataFrame(plot_rows)
    posterior_summary = pd.DataFrame(posterior_rows)
    model_diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostic_details = pd.concat(diagnostic_detail_tables, ignore_index=True)

    return plot_results, posterior_summary, model_diagnostics, diagnostic_details


def subset_for_parameter(results: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Return rows for one parameter in the requested dimension order."""
    subset = results[results["Parameter"] == parameter].copy()
    subset["Dimension"] = pd.Categorical(
        subset["Dimension"], categories=DIMENSIONS, ordered=True
    )
    return subset.sort_values("Dimension")


def draw_horizontal_interval_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    xlabel: str,
    title: str,
    xlim: tuple[float, float],
    xticks: list[float],
    show_ylabels: bool = False,
    percent_axis: bool = False,
    reference_zero: bool = False,
    rope: tuple[float, float] | None = None,
) -> None:
    """Draw a compact posterior mean and 95% HDI panel."""
    y_positions = np.array([2, 1, 0])
    y_lookup = dict(zip(DIMENSIONS, y_positions))

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.92", linewidth=0.6, linestyle="-", zorder=0)

    if rope is not None:
        ax.axvspan(
            rope[0], rope[1], color="0.65", alpha=0.12, linewidth=0, zorder=0
        )
        ax.axvline(
            rope[0],
            color="0.55",
            linestyle=(0, (1.5, 2.5)),
            linewidth=0.8,
            zorder=1,
        )
        ax.axvline(
            rope[1],
            color="0.55",
            linestyle=(0, (1.5, 2.5)),
            linewidth=0.8,
            zorder=1,
        )

    if reference_zero:
        ax.axvline(
            0,
            color="0.25",
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            zorder=1,
        )

    for _, row in data.iterrows():
        dimension = str(row["Dimension"])
        y = y_lookup[dimension]
        mean = float(row["Posterior_mean"])
        lower = float(row["HDI_95_lower"])
        upper = float(row["HDI_95_upper"])

        xerr = np.array([[mean - lower], [upper - mean]])

        ax.errorbar(
            mean,
            y,
            xerr=xerr,
            fmt="o",
            markersize=5.8,
            markerfacecolor=COLORS[dimension],
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=COLORS[dimension],
            elinewidth=1.7,
            capsize=3.0,
            capthick=1.1,
            zorder=3,
        )

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.55, 2.55)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="semibold", pad=8)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_yticks(y_positions)

    if show_ylabels:
        ax.set_yticklabels(DIMENSIONS)
        ax.tick_params(axis="y", length=3, pad=5)
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
    """Create a publication-ready three-panel Bayesian summary figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    mu_data = subset_for_parameter(results, "mu_LLM")
    sigma_data = subset_for_parameter(results, "sigma_LLM")
    tie_data = subset_for_parameter(results, "tie_probability")

    fig = plt.figure(figsize=(7.2, 2.35))
    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.18, 1.0, 1.0],
        wspace=0.42,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
    ]

    draw_horizontal_interval_panel(
        axes[0],
        mu_data,
        xlabel=r"$\mu_{\mathrm{LLM}}$",
        title="Overall source effect",
        xlim=(-0.25, 0.25),
        xticks=[-0.2, -0.1, 0.0, 0.1, 0.2],
        show_ylabels=True,
        reference_zero=True,
        rope=(ROPE_LOW, ROPE_HIGH),
    )

    draw_horizontal_interval_panel(
        axes[1],
        sigma_data,
        xlabel=r"$\sigma_{\mathrm{LLM}}$",
        title="Between-expert heterogeneity",
        xlim=(0.00, 0.45),
        xticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        show_ylabels=False,
    )

    draw_horizontal_interval_panel(
        axes[2],
        tie_data,
        xlabel=r"$P(\mathrm{Tie}\mid\Delta=0)$",
        title="Baseline tie probability",
        xlim=(0.00, 0.30),
        xticks=[0.0, 0.1, 0.2, 0.3],
        show_ylabels=False,
        percent_axis=True,
    )

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.16,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.25, top=0.84)
    fig.savefig(SCRIPT_DIR / f"{OUTPUT_STEM}.png", dpi=600, bbox_inches="tight")
    fig.savefig(SCRIPT_DIR / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)


def _dimension_row(summary: pd.DataFrame, dimension: str) -> pd.Series:
    row = summary.loc[summary["Dimension"] == dimension]
    if len(row) != 1:
        raise ValueError(f"Expected one posterior-summary row for {dimension}.")
    return row.iloc[0]


def build_manuscript_text(
    posterior_summary: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Generate ready-to-copy manuscript text from the calculated values."""
    novelty = _dimension_row(posterior_summary, "Novelty")
    significance = _dimension_row(posterior_summary, "Significance")
    testability = _dimension_row(posterior_summary, "Testability")

    total_divergences = int(model_diagnostics["Divergent_transitions"].sum())
    max_rhat = float(model_diagnostics["Maximum_R_hat"].max())
    min_bulk = float(model_diagnostics["Minimum_ESS_bulk"].min())
    min_tail = float(model_diagnostics["Minimum_ESS_tail"].min())

    if total_divergences == 0:
        convergence_sentence = (
            "All three Bayesian hierarchical Davidson–Bradley–Terry models "
            "converged without divergent transitions."
        )
    else:
        convergence_sentence = (
            "The three Bayesian hierarchical Davidson–Bradley–Terry models "
            f"contained {total_divergences:,} divergent transitions in total."
        )

    paragraph_1 = (
        f"{convergence_sentence} The maximum R-hat was {max_rhat:.3f}, and the "
        f"minimum bulk and tail effective sample sizes were {min_bulk:,.0f} and "
        f"{min_tail:,.0f}, respectively. Population-level source effects were close "
        f"to zero (Fig. 2a). For novelty, the posterior mean was "
        f"{novelty['mu_LLM_mean']:.3f} (95% highest-density interval (HDI), "
        f"{novelty['mu_LLM_HDI95_lower']:.3f} to "
        f"{novelty['mu_LLM_HDI95_upper']:.3f}). The corresponding means were "
        f"{significance['mu_LLM_mean']:.3f} for significance (95% HDI, "
        f"{significance['mu_LLM_HDI95_lower']:.3f} to "
        f"{significance['mu_LLM_HDI95_upper']:.3f}) and "
        f"{testability['mu_LLM_mean']:.3f} for testability (95% HDI, "
        f"{testability['mu_LLM_HDI95_lower']:.3f} to "
        f"{testability['mu_LLM_HDI95_upper']:.3f}). All intervals included zero, "
        "and substantial posterior mass fell within the prespecified region of "
        "practical equivalence."
    )

    sigma_values = {
        dimension: float(_dimension_row(posterior_summary, dimension)["sigma_LLM_mean"])
        for dimension in DIMENSIONS
    }
    highest_sigma_dimension = max(sigma_values, key=sigma_values.get)

    paragraph_2 = (
        f"The posterior mean of between-expert heterogeneity was highest for "
        f"{highest_sigma_dimension.lower()}. The posterior mean between-expert "
        f"standard deviation was {novelty['sigma_LLM_mean']:.3f} for novelty, "
        f"{significance['sigma_LLM_mean']:.3f} for significance and "
        f"{testability['sigma_LLM_mean']:.3f} for testability (Fig. 2b). At equal "
        f"latent preference, model-implied tie probabilities were "
        f"{novelty['tie_probability_mean'] * 100:.1f}%, "
        f"{significance['tie_probability_mean'] * 100:.1f}% and "
        f"{testability['tie_probability_mean'] * 100:.1f}% for novelty, significance "
        "and testability, respectively (Fig. 2c). The descriptive and model-based "
        "results therefore provided no strong evidence of a source-level advantage, "
        "while showing variation in expert decision behavior."
    )

    rope_sentence = (
        "The posterior probabilities within the ROPE were "
        f"{novelty['P_mu_within_ROPE'] * 100:.1f}%, "
        f"{significance['P_mu_within_ROPE'] * 100:.1f}% and "
        f"{testability['P_mu_within_ROPE'] * 100:.1f}% for novelty, significance "
        "and testability, respectively."
    )

    return pd.DataFrame(
        [
            {"Section": "Paragraph 1", "Text": paragraph_1},
            {"Section": "Paragraph 2", "Text": paragraph_2},
            {"Section": "Optional ROPE sentence", "Text": rope_sentence},
        ]
    )


def build_overall_diagnostics(model_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Add one overall row summarizing diagnostics across all three models."""
    max_rhat_index = model_diagnostics["Maximum_R_hat"].idxmax()
    min_bulk_index = model_diagnostics["Minimum_ESS_bulk"].idxmin()
    min_tail_index = model_diagnostics["Minimum_ESS_tail"].idxmin()

    overall = {
        "Dimension": "Overall",
        "Source_file": "All three NetCDF files",
        "Chains": "—",
        "Draws_per_chain": "—",
        "Total_posterior_draws": int(model_diagnostics["Total_posterior_draws"].sum()),
        "Divergent_transitions": int(
            model_diagnostics["Divergent_transitions"].sum()
        ),
        "Reached_max_treedepth": (
            int(model_diagnostics["Reached_max_treedepth"].fillna(0).sum())
            if "Reached_max_treedepth" in model_diagnostics
            else None
        ),
        "Maximum_R_hat": float(model_diagnostics["Maximum_R_hat"].max()),
        "Maximum_R_hat_parameter": (
            f"{model_diagnostics.loc[max_rhat_index, 'Dimension']}: "
            f"{model_diagnostics.loc[max_rhat_index, 'Maximum_R_hat_parameter']}"
        ),
        "Minimum_ESS_bulk": float(model_diagnostics["Minimum_ESS_bulk"].min()),
        "Minimum_ESS_bulk_parameter": (
            f"{model_diagnostics.loc[min_bulk_index, 'Dimension']}: "
            f"{model_diagnostics.loc[min_bulk_index, 'Minimum_ESS_bulk_parameter']}"
        ),
        "Minimum_ESS_tail": float(model_diagnostics["Minimum_ESS_tail"].min()),
        "Minimum_ESS_tail_parameter": (
            f"{model_diagnostics.loc[min_tail_index, 'Dimension']}: "
            f"{model_diagnostics.loc[min_tail_index, 'Minimum_ESS_tail_parameter']}"
        ),
    }

    return pd.concat(
        [model_diagnostics, pd.DataFrame([overall])],
        ignore_index=True,
    )


def write_excel_report(
    posterior_summary: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    diagnostic_details: pd.DataFrame,
    plot_results: pd.DataFrame,
    manuscript_text: pd.DataFrame,
) -> None:
    """Write all numerical outputs and generated text to a formatted Excel workbook."""
    readme = pd.DataFrame(
        [
            {
                "Item": "Posterior_summary",
                "Description": (
                    "Posterior means, SDs and 95% HDIs for mu_LLM and sigma_LLM; "
                    "model-implied tie probabilities; directional and ROPE probabilities."
                ),
            },
            {
                "Item": "Model_diagnostics",
                "Description": (
                    "Chains, draws, divergent transitions, maximum R-hat and minimum "
                    "bulk/tail ESS for each model and across all models."
                ),
            },
            {
                "Item": "Diagnostic_details",
                "Description": (
                    "R-hat, bulk ESS and tail ESS for every scalar or indexed posterior parameter."
                ),
            },
            {
                "Item": "Plot_values",
                "Description": "Exact posterior means and 95% HDIs used in Fig. 2a–c.",
            },
            {
                "Item": "Manuscript_text",
                "Description": "Results paragraphs generated directly from the current NetCDF files.",
            },
            {
                "Item": "Calculation notes",
                "Description": (
                    "HDIs are computed with ArviZ. Tie probabilities are transformed at "
                    "delta=0 for each posterior draw before summarization. Maximum R-hat "
                    "and minimum ESS are calculated across all posterior variables, including "
                    "indexed expert-level parameters."
                ),
            },
        ]
    )

    overall_diagnostics = build_overall_diagnostics(model_diagnostics)

    try:
        writer = pd.ExcelWriter(EXCEL_PATH, engine="xlsxwriter")
    except ImportError as exc:
        raise ImportError(
            "Writing the formatted Excel report requires XlsxWriter. "
            "Install it with: pip install XlsxWriter"
        ) from exc

    with writer:
        posterior_summary.to_excel(
            writer, sheet_name="Posterior_summary", index=False, freeze_panes=(1, 2)
        )
        overall_diagnostics.to_excel(
            writer, sheet_name="Model_diagnostics", index=False, freeze_panes=(1, 2)
        )
        diagnostic_details.to_excel(
            writer, sheet_name="Diagnostic_details", index=False, freeze_panes=(1, 2)
        )
        plot_results.to_excel(
            writer, sheet_name="Plot_values", index=False, freeze_panes=(1, 2)
        )
        manuscript_text.to_excel(
            writer, sheet_name="Manuscript_text", index=False, freeze_panes=(1, 0)
        )
        readme.to_excel(writer, sheet_name="README", index=False, freeze_panes=(1, 0))

        workbook = writer.book
        header_format = workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )
        number_format = workbook.add_format({"num_format": "0.0000"})
        integer_format = workbook.add_format({"num_format": "0"})
        percent_format = workbook.add_format({"num_format": "0.0%"})
        wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})
        note_format = workbook.add_format(
            {"text_wrap": True, "valign": "top", "bg_color": "#F2F2F2"}
        )

        for sheet_name, dataframe in (
            ("Posterior_summary", posterior_summary),
            ("Model_diagnostics", overall_diagnostics),
            ("Diagnostic_details", diagnostic_details),
            ("Plot_values", plot_results),
            ("Manuscript_text", manuscript_text),
            ("README", readme),
        ):
            worksheet = writer.sheets[sheet_name]
            worksheet.autofilter(0, 0, len(dataframe), max(len(dataframe.columns) - 1, 0))
            worksheet.set_row(0, 24, header_format)

        posterior_ws = writer.sheets["Posterior_summary"]
        posterior_ws.set_column("A:A", 14)
        posterior_ws.set_column("B:B", 28)
        posterior_ws.set_column("C:C", 16, integer_format)
        posterior_ws.set_column("D:G", 18, number_format)
        posterior_ws.set_column("H:K", 18, percent_format)
        posterior_ws.set_column("L:S", 21, number_format)
        posterior_ws.set_column("Q:S", 24, percent_format)

        diagnostics_ws = writer.sheets["Model_diagnostics"]
        diagnostics_ws.set_column("A:A", 14)
        diagnostics_ws.set_column("B:B", 28)
        diagnostics_ws.set_column("C:G", 20)
        diagnostics_ws.set_column("H:H", 16, number_format)
        diagnostics_ws.set_column("I:I", 34)
        diagnostics_ws.set_column("J:J", 18, number_format)
        diagnostics_ws.set_column("K:K", 34)
        diagnostics_ws.set_column("L:L", 18, number_format)
        diagnostics_ws.set_column("M:M", 34)

        detail_ws = writer.sheets["Diagnostic_details"]
        detail_ws.set_column("A:A", 14)
        detail_ws.set_column("B:B", 28)
        detail_ws.set_column("C:C", 34)
        detail_ws.set_column("D:F", 16, number_format)

        plot_ws = writer.sheets["Plot_values"]
        plot_ws.set_column("A:B", 18)
        plot_ws.set_column("C:F", 18, number_format)

        manuscript_ws = writer.sheets["Manuscript_text"]
        manuscript_ws.set_column("A:A", 24)
        manuscript_ws.set_column("B:B", 140, wrap_format)
        for row_index in range(1, len(manuscript_text) + 1):
            manuscript_ws.set_row(row_index, 95)

        readme_ws = writer.sheets["README"]
        readme_ws.set_column("A:A", 24)
        readme_ws.set_column("B:B", 110, note_format)
        for row_index in range(1, len(readme) + 1):
            readme_ws.set_row(row_index, 52)


def main() -> None:
    plot_results, posterior_summary, model_diagnostics, diagnostic_details = (
        compute_results()
    )

    manuscript_text = build_manuscript_text(posterior_summary, model_diagnostics)

    plot_results.to_csv(CSV_PATH, index=False, float_format="%.10g")
    make_figure(plot_results)
    write_excel_report(
        posterior_summary,
        model_diagnostics,
        diagnostic_details,
        plot_results,
        manuscript_text,
    )

    print("\nSaved outputs")
    print("=" * 72)
    print(SCRIPT_DIR / f"{OUTPUT_STEM}.png")
    print(SCRIPT_DIR / f"{OUTPUT_STEM}.pdf")
    print(CSV_PATH)
    print(EXCEL_PATH)


if __name__ == "__main__":
    main()
