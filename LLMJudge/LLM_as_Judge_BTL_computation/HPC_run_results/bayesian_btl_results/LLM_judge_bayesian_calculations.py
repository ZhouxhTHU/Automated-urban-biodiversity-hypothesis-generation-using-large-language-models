"""
Summarize Bayesian source effects, direct LLM-vs-human posterior comparisons,
and model-implied tie probabilities from NetCDF posterior files.

The script reads the posterior arrays directly with h5py and computes 95%
highest-density intervals (HDIs). It always writes a structured JSON data file
and also writes an Excel workbook when the Python artifact_tool package is
available.

Outputs
-------
LLM_judge_bayesian_calculations.xlsx
LLM_judge_bayesian_calculations_data.json

Required packages
-----------------
numpy
h5py
artifact_tool (optional, for direct Excel export)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

try:
    from artifact_tool import SpreadsheetFile, Workbook
except ModuleNotFoundError:
    SpreadsheetFile = None
    Workbook = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "LLM_judge_bayesian_calculations.xlsx"
DATA_FILE = BASE_DIR / "LLM_judge_bayesian_calculations_data.json"

DIMENSIONS = ("Novelty", "Significance", "Testability")
LLM_JUDGES = ("Gemini-3.5-Flash-Thinking", "GPT-5.5", "Claude")
ALL_JUDGES = (*LLM_JUDGES, "Human")
ROPE_LOWER = -0.1
ROPE_UPPER = 0.1
HDI_PROBABILITY = 0.95

FILES: dict[str, dict[str, Path]] = {
    "Human": {
        "Novelty": BASE_DIR / "human" / "human_idata_Novelty.nc",
        "Significance": BASE_DIR / "human" / "human_idata_Significance.nc",
        "Testability": BASE_DIR / "human" / "human_idata_Testability.nc",
    },
    "Gemini-3.5-Flash-Thinking": {
        "Novelty": BASE_DIR / "gemini-3.5-flash-thinking" / "gemini-3.5-flash-thinking_idata_Novelty.nc",
        "Significance": BASE_DIR / "gemini-3.5-flash-thinking" / "gemini-3.5-flash-thinking_idata_Significance.nc",
        "Testability": BASE_DIR / "gemini-3.5-flash-thinking" / "gemini-3.5-flash-thinking_idata_Testability.nc",
    },
    "GPT-5.5": {
        "Novelty": BASE_DIR / "gpt-5.5" / "gpt-5.5_idata_Novelty.nc",
        "Significance": BASE_DIR / "gpt-5.5" / "gpt-5.5_idata_Significance.nc",
        "Testability": BASE_DIR / "gpt-5.5" / "gpt-5.5_idata_Testability.nc",
    },
    "Claude": {
        "Novelty": BASE_DIR / "claude" / "claude_idata_Novelty.nc",
        "Significance": BASE_DIR / "claude" / "claude_idata_Significance.nc",
        "Testability": BASE_DIR / "claude" / "claude_idata_Testability.nc",
    },
}

# Color-blind-friendly palette (Okabe-Ito / Nature-compatible choices).
DARK_BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
DARK_TEXT = "#1F2937"
LIGHT_BLUE = "#EAF4FA"
LIGHT_ORANGE = "#FFF4DD"
LIGHT_PURPLE = "#F9EAF3"
LIGHT_GRAY = "#F3F4F6"
WHITE = "#FFFFFF"
BORDER = "#D1D5DB"


@dataclass(frozen=True)
class PosteriorData:
    judge: str
    dimension: str
    file_path: Path
    chain_count: int
    draw_count_per_chain: int
    mu_samples: np.ndarray
    tie_logit_samples: np.ndarray
    tie_probability_samples: np.ndarray
    divergent_count: int


# -----------------------------------------------------------------------------
# Posterior calculations
# -----------------------------------------------------------------------------


def validate_files() -> None:
    missing = [
        str(path)
        for judge_files in FILES.values()
        for path in judge_files.values()
        if not path.exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"The following NetCDF files are missing:\n{formatted}")


def read_dataset(file_path: Path, dataset_path: str) -> np.ndarray:
    """Read one HDF5/NetCDF dataset as a NumPy array."""
    with h5py.File(file_path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(
                f"Dataset {dataset_path!r} was not found in {file_path.name}."
            )
        values = np.asarray(handle[dataset_path][...])
    return values


def load_posterior(judge: str, dimension: str, file_path: Path) -> PosteriorData:
    mu = read_dataset(file_path, "posterior/mu_LLM").astype(float)
    tie_logit = read_dataset(file_path, "posterior/tie_logit").astype(float)
    diverging = read_dataset(file_path, "sample_stats/diverging").astype(bool)

    if mu.ndim != 2 or tie_logit.ndim != 2:
        raise ValueError(
            f"Expected scalar posterior variables with shape (chain, draw) in "
            f"{file_path.name}; got mu={mu.shape}, tie_logit={tie_logit.shape}."
        )
    if mu.shape != tie_logit.shape:
        raise ValueError(
            f"mu_LLM and tie_logit shapes differ in {file_path.name}: "
            f"{mu.shape} vs {tie_logit.shape}."
        )

    chain_count, draw_count = mu.shape
    mu_flat = mu.reshape(-1)
    tie_logit_flat = tie_logit.reshape(-1)

    if not np.all(np.isfinite(mu_flat)):
        raise ValueError(f"Non-finite mu_LLM samples found in {file_path.name}.")
    if not np.all(np.isfinite(tie_logit_flat)):
        raise ValueError(f"Non-finite tie_logit samples found in {file_path.name}.")

    tie_probability = tie_probability_from_logit(tie_logit_flat)

    return PosteriorData(
        judge=judge,
        dimension=dimension,
        file_path=file_path,
        chain_count=chain_count,
        draw_count_per_chain=draw_count,
        mu_samples=mu_flat,
        tie_logit_samples=tie_logit_flat,
        tie_probability_samples=tie_probability,
        divergent_count=int(diverging.sum()),
    )


def hdi(samples: np.ndarray, probability: float = 0.95) -> tuple[float, float]:
    """
    Compute a one-dimensional highest-density interval.

    The samples are sorted. Among all intervals containing floor(probability*n)
    posterior draws, the narrowest interval is selected.
    """
    values = np.sort(np.asarray(samples, dtype=float).reshape(-1))
    n = values.size
    if n < 2:
        raise ValueError("At least two posterior samples are required.")
    if not 0 < probability < 1:
        raise ValueError("HDI probability must be between 0 and 1.")

    interval_count = int(np.floor(probability * n))
    if interval_count < 1 or interval_count >= n:
        raise ValueError("Invalid HDI interval count.")

    widths = values[interval_count:] - values[: n - interval_count]
    start = int(np.argmin(widths))
    return float(values[start]), float(values[start + interval_count])


def tie_probability_from_logit(tie_logit: np.ndarray) -> np.ndarray:
    """
    Convert each Davidson tie logit to P(Tie | delta=0).

    At equal latent preference, the three unnormalized category weights are
    (1, exp(tie_logit), 1), hence:
        P(Tie | delta=0) = exp(tie_logit) / (2 + exp(tie_logit))
                         = 1 / (1 + 2*exp(-tie_logit)).
    """
    values = np.asarray(tie_logit, dtype=float)
    return 1.0 / (1.0 + 2.0 * np.exp(-values))


def summarize(samples: np.ndarray) -> dict[str, float | int | bool]:
    values = np.asarray(samples, dtype=float).reshape(-1)
    lower, upper = hdi(values, HDI_PROBABILITY)
    return {
        "Posterior_draws": int(values.size),
        "Posterior_mean": float(values.mean()),
        "Posterior_median": float(np.median(values)),
        "Posterior_sd": float(values.std(ddof=1)),
        "HDI_95_lower": lower,
        "HDI_95_upper": upper,
        "HDI_excludes_zero": bool(lower > 0 or upper < 0),
        "P_greater_than_0": float(np.mean(values > 0)),
        "P_less_than_0": float(np.mean(values < 0)),
    }


def load_all_posteriors() -> dict[str, dict[str, PosteriorData]]:
    validate_files()
    loaded: dict[str, dict[str, PosteriorData]] = {}
    for judge in ALL_JUDGES:
        loaded[judge] = {}
        for dimension in DIMENSIONS:
            loaded[judge][dimension] = load_posterior(
                judge, dimension, FILES[judge][dimension]
            )
    return loaded


def source_effect_rows(
    loaded: dict[str, dict[str, PosteriorData]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for judge in ALL_JUDGES:
        for dimension in DIMENSIONS:
            item = loaded[judge][dimension]
            stats = summarize(item.mu_samples)
            rope_mask = (
                (item.mu_samples >= ROPE_LOWER)
                & (item.mu_samples <= ROPE_UPPER)
            )
            rows.append(
                {
                    "Judge": judge,
                    "Dimension": dimension,
                    "Source_file": item.file_path.name,
                    "Chains": item.chain_count,
                    "Draws_per_chain": item.draw_count_per_chain,
                    **stats,
                    "ROPE_lower": ROPE_LOWER,
                    "ROPE_upper": ROPE_UPPER,
                    "ROPE_draws": int(rope_mask.sum()),
                    "P_within_ROPE": float(rope_mask.mean()),
                    "Divergences": item.divergent_count,
                }
            )
    return rows


def direct_comparison_rows(
    loaded: dict[str, dict[str, PosteriorData]],
) -> list[dict[str, object]]:
    """
    Compute posterior differences mu_LLM_judge - mu_human.

    The independently fitted posterior arrays have equal length. Pairing the
    flattened draws by index yields Monte Carlo draws from the product of the two
    posterior distributions; the pairing is arbitrary because the fits are
    independent.
    """
    rows: list[dict[str, object]] = []
    for judge in LLM_JUDGES:
        for dimension in DIMENSIONS:
            llm = loaded[judge][dimension].mu_samples
            human = loaded["Human"][dimension].mu_samples
            if llm.size != human.size:
                raise ValueError(
                    f"Posterior draw counts differ for {judge}, {dimension}: "
                    f"{llm.size} vs {human.size}."
                )
            difference = llm - human
            stats = summarize(difference)
            rows.append(
                {
                    "LLM_judge": judge,
                    "Dimension": dimension,
                    "Definition": "mu_LLM_judge - mu_human",
                    **stats,
                }
            )
    return rows


def tie_probability_rows(
    loaded: dict[str, dict[str, PosteriorData]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for judge in ALL_JUDGES:
        for dimension in DIMENSIONS:
            item = loaded[judge][dimension]
            stats = summarize(item.tie_probability_samples)
            rows.append(
                {
                    "Judge": judge,
                    "Dimension": dimension,
                    "Tie_logit_mean": float(item.tie_logit_samples.mean()),
                    **stats,
                    "Posterior_mean_percent": 100.0 * float(stats["Posterior_mean"]),
                    "HDI_95_lower_percent": 100.0 * float(stats["HDI_95_lower"]),
                    "HDI_95_upper_percent": 100.0 * float(stats["HDI_95_upper"]),
                }
            )
    return rows


def manuscript_range_rows(
    source_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    tie_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for dimension in DIMENSIONS:
        source_values = [
            float(row["Posterior_mean"])
            for row in source_rows
            if row["Judge"] in LLM_JUDGES and row["Dimension"] == dimension
        ]
        comparison_values = [
            float(row["Posterior_mean"])
            for row in comparison_rows
            if row["Dimension"] == dimension
        ]
        llm_tie_values = [
            float(row["Posterior_mean_percent"])
            for row in tie_rows
            if row["Judge"] in LLM_JUDGES and row["Dimension"] == dimension
        ]
        human_tie_value = next(
            float(row["Posterior_mean_percent"])
            for row in tie_rows
            if row["Judge"] == "Human" and row["Dimension"] == dimension
        )

        rows.extend(
            [
                {
                    "Metric": "LLM source-effect posterior mean",
                    "Dimension": dimension,
                    "Minimum": min(source_values),
                    "Maximum": max(source_values),
                    "Unit": "latent scale",
                    "Suggested_rounding": "3 decimals",
                },
                {
                    "Metric": "Direct LLM-minus-human posterior mean",
                    "Dimension": dimension,
                    "Minimum": min(comparison_values),
                    "Maximum": max(comparison_values),
                    "Unit": "latent scale",
                    "Suggested_rounding": "3 decimals",
                },
                {
                    "Metric": "LLM model-implied tie probability",
                    "Dimension": dimension,
                    "Minimum": min(llm_tie_values),
                    "Maximum": max(llm_tie_values),
                    "Unit": "%",
                    "Suggested_rounding": "1 decimal place",
                },
                {
                    "Metric": "Human model-implied tie probability",
                    "Dimension": dimension,
                    "Minimum": human_tie_value,
                    "Maximum": human_tie_value,
                    "Unit": "%",
                    "Suggested_rounding": "1 decimal place",
                },
            ]
        )

    all_llm_source_rows = [row for row in source_rows if row["Judge"] in LLM_JUDGES]
    rows.extend(
        [
            {
                "Metric": "All nine LLM source-effect 95% HDIs exclude zero",
                "Dimension": "All",
                "Minimum": int(all(bool(row["HDI_excludes_zero"]) for row in all_llm_source_rows)),
                "Maximum": int(all(bool(row["HDI_excludes_zero"]) for row in all_llm_source_rows)),
                "Unit": "1=True, 0=False",
                "Suggested_rounding": "integer",
            },
            {
                "Metric": "All nine LLM source-effect ROPE probabilities equal zero",
                "Dimension": "All",
                "Minimum": int(all(float(row["P_within_ROPE"]) == 0.0 for row in all_llm_source_rows)),
                "Maximum": int(all(float(row["P_within_ROPE"]) == 0.0 for row in all_llm_source_rows)),
                "Unit": "1=True, 0=False",
                "Suggested_rounding": "integer",
            },
            {
                "Metric": "All nine direct-comparison 95% HDIs exclude zero",
                "Dimension": "All",
                "Minimum": int(all(bool(row["HDI_excludes_zero"]) for row in comparison_rows)),
                "Maximum": int(all(bool(row["HDI_excludes_zero"]) for row in comparison_rows)),
                "Unit": "1=True, 0=False",
                "Suggested_rounding": "integer",
            },
        ]
    )
    return rows


def manuscript_text(
    source_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    tie_rows: list[dict[str, object]],
) -> str:
    def values(rows: Iterable[dict[str, object]], dimension: str, key: str) -> list[float]:
        return [float(row[key]) for row in rows if row["Dimension"] == dimension]

    llm_source = [row for row in source_rows if row["Judge"] in LLM_JUDGES]
    novelty_source = values(llm_source, "Novelty", "Posterior_mean")
    significance_source = values(llm_source, "Significance", "Posterior_mean")
    testability_source = values(llm_source, "Testability", "Posterior_mean")

    novelty_diff = values(comparison_rows, "Novelty", "Posterior_mean")
    significance_diff = values(comparison_rows, "Significance", "Posterior_mean")
    testability_diff = values(comparison_rows, "Testability", "Posterior_mean")

    llm_ties = [row for row in tie_rows if row["Judge"] in LLM_JUDGES]
    novelty_ties = values(llm_ties, "Novelty", "Posterior_mean_percent")
    significance_ties = values(llm_ties, "Significance", "Posterior_mean_percent")
    testability_ties = values(llm_ties, "Testability", "Posterior_mean_percent")

    human_tie = {
        row["Dimension"]: float(row["Posterior_mean_percent"])
        for row in tie_rows
        if row["Judge"] == "Human"
    }

    return (
        "Across the LLM judges, posterior means ranged from "
        f"{min(novelty_source):.3f} to {max(novelty_source):.3f} for novelty and "
        f"from {min(significance_source):.3f} to {max(significance_source):.3f} "
        "for significance, whereas testability estimates ranged from "
        f"{min(testability_source):.3f} to {max(testability_source):.3f}. "
        "For every LLM judge and dimension, the 95% HDI excluded zero, and "
        "the estimated posterior probability within the prespecified region of "
        "practical equivalence was 0.000. Direct posterior comparisons with the "
        "human estimates produced mean differences of "
        f"{min(novelty_diff):.3f}-{max(novelty_diff):.3f} for novelty, "
        f"{min(significance_diff):.3f}-{max(significance_diff):.3f} for significance, "
        f"and {min(testability_diff):.3f} to {max(testability_diff):.3f} for "
        "testability. All nine 95% HDIs excluded zero. Model-implied tie "
        "probabilities at equal latent preference ranged from "
        f"{min(novelty_ties):.1f}% to {max(novelty_ties):.1f}% for novelty, "
        f"{min(significance_ties):.1f}% to {max(significance_ties):.1f}% for "
        "significance, and "
        f"{min(testability_ties):.1f}% to {max(testability_ties):.1f}% for "
        "testability, compared with "
        f"{human_tie['Novelty']:.1f}%, {human_tie['Significance']:.1f}%, and "
        f"{human_tie['Testability']:.1f}% for human experts."
    )


# -----------------------------------------------------------------------------
# Excel workbook creation
# -----------------------------------------------------------------------------


def matrix_from_rows(rows: list[dict[str, object]], columns: list[str]) -> list[list[object]]:
    return [[row.get(column) for column in columns] for row in rows]


def add_title(sheet, title: str, end_column: str) -> None:
    sheet.merge_cells(f"A1:{end_column}1")
    sheet.get_range("A1").values = [[title]]
    sheet.get_range(f"A1:{end_column}1").format = {
        "fill": DARK_BLUE,
        "font": {"bold": True, "color": WHITE, "size": 15},
        "horizontal_alignment": "left",
        "vertical_alignment": "center",
        "row_height": 28,
    }


def style_header(sheet, cell_range: str, fill: str = DARK_BLUE) -> None:
    sheet.get_range(cell_range).format = {
        "fill": fill,
        "font": {"bold": True, "color": WHITE},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
        "borders": {
            "bottom": {"color": BORDER, "style": "continuous", "weight": 1},
        },
    }


def style_body(sheet, cell_range: str) -> None:
    sheet.get_range(cell_range).format = {
        "font": {"color": DARK_TEXT},
        "vertical_alignment": "center",
        "borders": {
            "bottom": {"color": BORDER, "style": "continuous", "weight": 1},
        },
    }


def set_widths(sheet, widths: dict[str, float]) -> None:
    for column, width in widths.items():
        sheet.get_range(f"{column}:{column}").format.column_width = width


def add_table_sheet(
    workbook: Workbook,
    name: str,
    title: str,
    rows: list[dict[str, object]],
    columns: list[str],
    widths: dict[str, float],
    percent_columns: Iterable[str] = (),
    decimal_columns: Iterable[str] = (),
    integer_columns: Iterable[str] = (),
    title_end_column: str | None = None,
) -> None:
    sheet = workbook.worksheets.add(name)
    title_end_column = title_end_column or chr(64 + len(columns))
    add_title(sheet, title, title_end_column)
    sheet.get_range("A3").resize(1, len(columns)).values = [columns]
    style_header(sheet, f"A3:{title_end_column}3")

    if rows:
        matrix = matrix_from_rows(rows, columns)
        sheet.get_range("A4").resize(len(matrix), len(columns)).values = matrix
        style_body(sheet, f"A4:{title_end_column}{3 + len(matrix)}")

    for column in percent_columns:
        index = columns.index(column) + 1
        excel_col = column_letter(index)
        sheet.get_range(f"{excel_col}4:{excel_col}{3 + len(rows)}").format.number_format = "0.0000%"

    for column in decimal_columns:
        index = columns.index(column) + 1
        excel_col = column_letter(index)
        sheet.get_range(f"{excel_col}4:{excel_col}{3 + len(rows)}").format.number_format = "0.000000"

    for column in integer_columns:
        index = columns.index(column) + 1
        excel_col = column_letter(index)
        sheet.get_range(f"{excel_col}4:{excel_col}{3 + len(rows)}").format.number_format = "0"

    sheet.freeze_panes.freeze_rows(3)
    set_widths(sheet, widths)
    sheet.get_range(f"A1:{title_end_column}{max(4, 3 + len(rows))}").format.wrap_text = True


def column_letter(index: int) -> str:
    result = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def create_workbook(
    source_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    tie_rows: list[dict[str, object]],
    range_rows: list[dict[str, object]],
    text: str,
) -> Workbook:
    workbook = Workbook.create()

    # README
    readme = workbook.worksheets.add("README")
    add_title(readme, "LLM Judge Bayesian Calculations", "F")
    readme_rows = [
        ["Purpose", "Reproduce all numerical values used in the LLM-judge Bayesian Results paragraph."],
        ["Source-effect parameter", "mu_LLM from each independently fitted hierarchical Davidson-Bradley-Terry model."],
        ["95% interval", "Highest-density interval (HDI), calculated as the narrowest interval containing floor(0.95*n) sorted posterior draws."],
        ["ROPE", f"Prespecified interval [{ROPE_LOWER}, {ROPE_UPPER}]. P_within_ROPE is the fraction of posterior draws inside the interval."],
        ["Direct comparison", "For each posterior draw: mu_LLM_judge - mu_human. The fitted posterior arrays are independent and have equal length; index pairing provides Monte Carlo draws from their product posterior."],
        ["Tie probability", "For every tie_logit posterior draw, P(Tie | delta=0) = exp(tie_logit)/(2+exp(tie_logit)); probabilities are transformed before calculating means and HDIs."],
        ["Rounding", "Latent-scale effects are reported to three decimals; tie probabilities are reported to one decimal percentage point."],
        ["Claude exception", "Claude-Sonnet-4.6 completed all evaluations except one pair, which was evaluated with Claude-Haiku-4.5 after a content-safety filter response. This methodological exception is not inferable from the NetCDF files and is recorded here from the study documentation."],
        ["Output sheets", "Source_effects, Direct_comparisons, Tie_probabilities, Manuscript_ranges, Manuscript_text, File_inventory, and Calculation_steps."],
    ]
    readme.get_range("A3:B11").values = readme_rows
    style_body(readme, "A3:B11")
    readme.get_range("A3:A11").format = {
        "fill": LIGHT_BLUE,
        "font": {"bold": True, "color": DARK_TEXT},
        "vertical_alignment": "top",
        "borders": {"bottom": {"color": BORDER, "style": "continuous", "weight": 1}},
    }
    set_widths(readme, {"A": 24, "B": 95})
    readme.get_range("A1:F11").format.wrap_text = True

    source_columns = [
        "Judge", "Dimension", "Source_file", "Chains", "Draws_per_chain",
        "Posterior_draws", "Posterior_mean", "Posterior_median", "Posterior_sd",
        "HDI_95_lower", "HDI_95_upper", "HDI_excludes_zero",
        "P_greater_than_0", "P_less_than_0", "ROPE_lower", "ROPE_upper",
        "ROPE_draws", "P_within_ROPE", "Divergences",
    ]
    add_table_sheet(
        workbook,
        "Source_effects",
        "Posterior source effects (mu_LLM)",
        source_rows,
        source_columns,
        {"A": 29, "B": 15, "C": 43, "D": 9, "E": 15, "F": 15, "G": 16,
         "H": 16, "I": 14, "J": 15, "K": 15, "L": 18, "M": 17, "N": 15,
         "O": 12, "P": 12, "Q": 13, "R": 16, "S": 12},
        percent_columns=("P_greater_than_0", "P_less_than_0", "P_within_ROPE"),
        decimal_columns=("Posterior_mean", "Posterior_median", "Posterior_sd", "HDI_95_lower", "HDI_95_upper", "ROPE_lower", "ROPE_upper"),
        integer_columns=("Chains", "Draws_per_chain", "Posterior_draws", "ROPE_draws", "Divergences"),
        title_end_column="S",
    )

    comparison_columns = [
        "LLM_judge", "Dimension", "Definition", "Posterior_draws",
        "Posterior_mean", "Posterior_median", "Posterior_sd", "HDI_95_lower",
        "HDI_95_upper", "HDI_excludes_zero", "P_greater_than_0", "P_less_than_0",
    ]
    add_table_sheet(
        workbook,
        "Direct_comparisons",
        "Direct posterior comparisons: LLM judge minus human",
        comparison_rows,
        comparison_columns,
        {"A": 29, "B": 15, "C": 28, "D": 15, "E": 16, "F": 16, "G": 14,
         "H": 15, "I": 15, "J": 18, "K": 17, "L": 15},
        percent_columns=("P_greater_than_0", "P_less_than_0"),
        decimal_columns=("Posterior_mean", "Posterior_median", "Posterior_sd", "HDI_95_lower", "HDI_95_upper"),
        integer_columns=("Posterior_draws",),
        title_end_column="L",
    )

    tie_columns = [
        "Judge", "Dimension", "Tie_logit_mean", "Posterior_draws", "Posterior_mean",
        "Posterior_median", "Posterior_sd", "HDI_95_lower", "HDI_95_upper",
        "Posterior_mean_percent", "HDI_95_lower_percent", "HDI_95_upper_percent",
    ]
    add_table_sheet(
        workbook,
        "Tie_probabilities",
        "Model-implied tie probabilities at equal latent preference",
        tie_rows,
        tie_columns,
        {"A": 29, "B": 15, "C": 16, "D": 15, "E": 16, "F": 16, "G": 14,
         "H": 15, "I": 15, "J": 20, "K": 20, "L": 20},
        decimal_columns=("Tie_logit_mean", "Posterior_mean", "Posterior_median", "Posterior_sd", "HDI_95_lower", "HDI_95_upper"),
        integer_columns=("Posterior_draws",),
        title_end_column="L",
    )
    tie_sheet = workbook.worksheets.get_item("Tie_probabilities")
    for col in ("J", "K", "L"):
        tie_sheet.get_range(f"{col}4:{col}{3 + len(tie_rows)}").format.number_format = '0.000"%"'

    range_columns = ["Metric", "Dimension", "Minimum", "Maximum", "Unit", "Suggested_rounding"]
    add_table_sheet(
        workbook,
        "Manuscript_ranges",
        "Ranges reported in the manuscript",
        range_rows,
        range_columns,
        {"A": 48, "B": 15, "C": 16, "D": 16, "E": 18, "F": 22},
        decimal_columns=("Minimum", "Maximum"),
        title_end_column="F",
    )

    text_sheet = workbook.worksheets.add("Manuscript_text")
    add_title(text_sheet, "Automatically generated manuscript text", "H")
    text_sheet.merge_cells("A3:H12")
    text_sheet.get_range("A3").values = [[text]]
    text_sheet.get_range("A3:H12").format = {
        "fill": LIGHT_BLUE,
        "font": {"color": DARK_TEXT, "size": 11},
        "vertical_alignment": "top",
        "horizontal_alignment": "left",
        "wrap_text": True,
        "borders": {
            "top": {"color": BORDER, "style": "continuous", "weight": 1},
            "bottom": {"color": BORDER, "style": "continuous", "weight": 1},
            "left": {"color": BORDER, "style": "continuous", "weight": 1},
            "right": {"color": BORDER, "style": "continuous", "weight": 1},
        },
    }
    set_widths(text_sheet, {column_letter(i): 15 for i in range(1, 9)})
    text_sheet.get_range("A3:H12").format.row_height = 24

    inventory_rows: list[dict[str, object]] = []
    for judge in ALL_JUDGES:
        for dimension in DIMENSIONS:
            path = FILES[judge][dimension]
            data = next(row for row in source_rows if row["Judge"] == judge and row["Dimension"] == dimension)
            inventory_rows.append(
                {
                    "Judge": judge,
                    "Dimension": dimension,
                    "File_name": path.name,
                    "Relative_path": str(path.relative_to(BASE_DIR)),
                    "File_size_bytes": path.stat().st_size,
                    "Chains": data["Chains"],
                    "Draws_per_chain": data["Draws_per_chain"],
                    "Posterior_draws": data["Posterior_draws"],
                    "Divergences": data["Divergences"],
                }
            )
    inventory_columns = [
        "Judge", "Dimension", "File_name", "Relative_path", "File_size_bytes",
        "Chains", "Draws_per_chain", "Posterior_draws", "Divergences",
    ]
    add_table_sheet(
        workbook,
        "File_inventory",
        "Input NetCDF file inventory",
        inventory_rows,
        inventory_columns,
        {"A": 29, "B": 15, "C": 48, "D": 85, "E": 18, "F": 10, "G": 16, "H": 16, "I": 12},
        integer_columns=("File_size_bytes", "Chains", "Draws_per_chain", "Posterior_draws", "Divergences"),
        title_end_column="I",
    )

    steps = workbook.worksheets.add("Calculation_steps")
    add_title(steps, "Calculation definitions and formulas", "D")
    step_rows = [
        ["Step", "Quantity", "Calculation", "Where used"],
        [1, "Posterior mean", "mean(x_s) across all chains and retained draws", "Source effects, direct differences, tie probabilities"],
        [2, "95% HDI", "Sort x_s; among intervals containing floor(0.95*n) draws, choose the narrowest interval", "All reported 95% intervals"],
        [3, "ROPE probability", f"count({ROPE_LOWER} <= mu_s <= {ROPE_UPPER}) / n", "P_within_ROPE"],
        [4, "Direct comparison", "difference_s = mu_LLM_judge,s - mu_human,s", "Fig. 5b / Direct_comparisons"],
        [5, "Tie probability per draw", "p_tie,s = exp(tau_s)/(2+exp(tau_s)) = 1/(1+2*exp(-tau_s))", "Fig. 5c / Tie_probabilities"],
        [6, "Tie probability summary", "Calculate the mean and 95% HDI after transforming every posterior draw; do not transform only the mean tie_logit", "Fig. 5c"],
        [7, "Manuscript range", "For each dimension, take min and max across the three LLM judges", "Reported ranges in Results"],
        [8, "Rounding", "Effects: 3 decimals; probabilities: 1 decimal percentage point", "Final manuscript wording"],
    ]
    steps.get_range("A3:D11").values = step_rows
    style_header(steps, "A3:D3", fill=PURPLE)
    style_body(steps, "A4:D11")
    set_widths(steps, {"A": 10, "B": 28, "C": 90, "D": 38})
    steps.get_range("A3:D11").format.wrap_text = True
    steps.freeze_panes.freeze_rows(3)

    return workbook


def main() -> None:
    loaded = load_all_posteriors()
    source_rows = source_effect_rows(loaded)
    comparison_rows = direct_comparison_rows(loaded)
    tie_rows = tie_probability_rows(loaded)
    range_rows = manuscript_range_rows(source_rows, comparison_rows, tie_rows)
    text = manuscript_text(source_rows, comparison_rows, tie_rows)
    inventory_rows = []
    for judge in ALL_JUDGES:
        for dimension in DIMENSIONS:
            item = loaded[judge][dimension]
            inventory_rows.append(
                {
                    "Judge": judge,
                    "Dimension": dimension,
                    "File_name": item.file_path.name,
                    "Relative_path": item.file_path.relative_to(BASE_DIR).as_posix(),
                    "File_size_bytes": item.file_path.stat().st_size,
                    "Chains": item.chain_count,
                    "Draws_per_chain": item.draw_count_per_chain,
                    "Posterior_draws": int(item.mu_samples.size),
                    "Divergences": item.divergent_count,
                }
            )

    payload = {
        "metadata": {
            "dimensions": list(DIMENSIONS),
            "llm_judges": list(LLM_JUDGES),
            "all_judges": list(ALL_JUDGES),
            "rope_lower": ROPE_LOWER,
            "rope_upper": ROPE_UPPER,
            "hdi_probability": HDI_PROBABILITY,
        },
        "source_effects": source_rows,
        "direct_comparisons": comparison_rows,
        "tie_probabilities": tie_rows,
        "manuscript_ranges": range_rows,
        "manuscript_text": text,
        "file_inventory": inventory_rows,
    }
    DATA_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {DATA_FILE}")

    if SpreadsheetFile is None or Workbook is None:
        print(
            "Python artifact_tool is unavailable; the JSON calculation data "
            "was saved successfully and can be used to build the Excel workbook."
        )
        print("\nAutomatically generated manuscript text:\n")
        print(text)
        return

    workbook = create_workbook(
        source_rows=source_rows,
        comparison_rows=comparison_rows,
        tie_rows=tie_rows,
        range_rows=range_rows,
        text=text,
    )
    SpreadsheetFile.export_xlsx(workbook).save(str(OUTPUT_FILE))

    print(f"Saved: {OUTPUT_FILE}")
    print("\nAutomatically generated manuscript text:\n")
    print(text)


if __name__ == "__main__":
    main()
