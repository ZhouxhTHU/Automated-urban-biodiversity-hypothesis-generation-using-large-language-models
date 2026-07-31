"""
Compare LLM-as-Judge outputs against human expert ratings.

This script aligns rows by expert, pair, and dimension, then reports direct
agreement, bias, tie behavior, pair-type breakdowns, and aggregate score
correlations for each LLM run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLS = ["Expert_ID", "Hypothesis_A_ID", "Hypothesis_B_ID", "Dimension"]
OUTCOME_ORDER = ["A_wins", "B_wins", "Tie"]
PAIR_TYPE_ORDER = ["HUM-vs-HUM", "HUM-vs-LLM", "LLM-vs-LLM"]


@dataclass(frozen=True)
class JudgeRun:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze consistency between human expert ratings and LLM-as-Judge ratings."
    )
    parser.add_argument(
        "--human-results",
        type=Path,
        default=Path("All_Comparison_Results.xlsx"),
        help="Path to human comparison results workbook.",
    )
    parser.add_argument(
        "--llm-run",
        action="append",
        nargs=2,
        metavar=("LABEL", "COMPARISON_RESULTS_XLSX"),
        help="LLM run label and path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/llm_human_consistency"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--rope",
        type=float,
        nargs=2,
        default=(-0.1, 0.1),
        metavar=("LOW", "HIGH"),
        help="Reserved for later model-layer analysis; recorded in metadata.",
    )
    return parser.parse_args()


def default_llm_runs() -> list[JudgeRun]:
    candidates = [
        JudgeRun("claude", Path("LLMJudge_results/comparison_results_claude.xlsx")),
        JudgeRun("gemini", Path("LLMJudge_results/comparison_results_gemini.xlsx")),
        JudgeRun("gpt", Path("LLMJudge_results/comparison_results_gpt.xlsx")),
        JudgeRun(
            "gemini-3.5-flash-thinking",
            Path("LLMJudge_results/gemini-3.5-flash-thinking/full_run_20260706_225626/comparison_results.xlsx"),
        ),
        JudgeRun(
            "gpt-5.5",
            Path("LLMJudge_results/gpt-5.5/full_run_20260707_123356/comparison_results.xlsx"),
        ),
    ]
    return [run for run in candidates if run.path.exists()]


def read_results(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {source} results file: {path}")
    df = pd.read_excel(path, sheet_name=0, dtype=str)
    missing = [col for col in KEY_COLS + ["Result"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df[KEY_COLS + ["Result"]].copy()
    for col in KEY_COLS + ["Result"]:
        df[col] = df[col].astype(str).str.strip()
    df["source"] = source
    df["outcome"] = df.apply(map_result_to_outcome, axis=1)
    df["pair_type"] = df.apply(pair_type, axis=1)
    return df


def hypothesis_type(hypothesis_id: str) -> str:
    text = str(hypothesis_id).strip().upper()
    if text.startswith("LLM") or text.startswith("HL"):
        return "LLM"
    if text.startswith("HUM") or text.startswith("HH"):
        return "HUM"
    return "UNKNOWN"


def pair_type(row: pd.Series) -> str:
    a_type = hypothesis_type(row["Hypothesis_A_ID"])
    b_type = hypothesis_type(row["Hypothesis_B_ID"])
    if {a_type, b_type} == {"HUM", "LLM"}:
        return "HUM-vs-LLM"
    if a_type == "HUM" and b_type == "HUM":
        return "HUM-vs-HUM"
    if a_type == "LLM" and b_type == "LLM":
        return "LLM-vs-LLM"
    return f"{a_type}-vs-{b_type}"


def map_result_to_outcome(row: pd.Series) -> str:
    result = str(row["Result"]).strip()
    if result.lower() == "tie":
        return "Tie"
    if result == str(row["Hypothesis_A_ID"]).strip():
        return "A_wins"
    if result == str(row["Hypothesis_B_ID"]).strip():
        return "B_wins"
    raise ValueError(
        "Invalid Result value "
        f"{result!r} for {row['Expert_ID']} {row['Hypothesis_A_ID']} vs {row['Hypothesis_B_ID']} "
        f"on {row['Dimension']}"
    )


def cohen_kappa(left: Iterable[str], right: Iterable[str], categories: list[str] = OUTCOME_ORDER) -> float:
    left = pd.Series(list(left), dtype="object")
    right = pd.Series(list(right), dtype="object")
    n = len(left)
    if n == 0:
        return np.nan
    observed = float((left == right).mean())
    pe = 0.0
    for cat in categories:
        pe += float((left == cat).mean() * (right == cat).mean())
    if np.isclose(1.0 - pe, 0.0):
        return np.nan
    return (observed - pe) / (1.0 - pe)


def gwet_ac1(left: Iterable[str], right: Iterable[str], categories: list[str] = OUTCOME_ORDER) -> float:
    left = pd.Series(list(left), dtype="object")
    right = pd.Series(list(right), dtype="object")
    n = len(left)
    k = len(categories)
    if n == 0 or k <= 1:
        return np.nan
    observed = float((left == right).mean())
    ratings = pd.concat([left, right], ignore_index=True)
    p = np.array([(ratings == cat).mean() for cat in categories], dtype=float)
    chance = float(np.sum(p * (1.0 - p)) / (k - 1))
    if np.isclose(1.0 - chance, 0.0):
        return np.nan
    return (observed - chance) / (1.0 - chance)


def weighted_agreement(
    left: pd.Series,
    right: pd.Series,
    categories: list[str],
    weights: np.ndarray,
) -> float:
    idx = {cat: i for i, cat in enumerate(categories)}
    total = 0.0
    for a, b in zip(left, right):
        total += weights[idx[a], idx[b]]
    return total / len(left) if len(left) else np.nan


def gwet_ac2_quadratic(left: Iterable[str], right: Iterable[str]) -> float:
    categories = ["B_wins", "Tie", "A_wins"]
    left = pd.Series(list(left), dtype="object")
    right = pd.Series(list(right), dtype="object")
    n = len(left)
    k = len(categories)
    if n == 0 or k <= 1:
        return np.nan
    weights = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            weights[i, j] = 1.0 - ((i - j) ** 2 / ((k - 1) ** 2))
    observed = weighted_agreement(left, right, categories, weights)
    ratings = pd.concat([left, right], ignore_index=True)
    p = np.array([(ratings == cat).mean() for cat in categories], dtype=float)
    chance = 0.0
    for i in range(k):
        for j in range(k):
            chance += weights[i, j] * p[i] * p[j]
    if np.isclose(1.0 - chance, 0.0):
        return np.nan
    return (observed - chance) / (1.0 - chance)


def summarize_group(group: pd.DataFrame, group_cols: list[str]) -> dict[str, object]:
    human = group["human_outcome"]
    llm = group["llm_outcome"]
    n = len(group)
    exact = float((human == llm).mean()) if n else np.nan
    row = {
        "n": n,
        "exact_agreement": exact,
        "cohen_kappa": cohen_kappa(human, llm),
        "gwet_ac1": gwet_ac1(human, llm),
        "gwet_ac2_quadratic": gwet_ac2_quadratic(human, llm),
        "human_tie_rate": float((human == "Tie").mean()) if n else np.nan,
        "llm_tie_rate": float((llm == "Tie").mean()) if n else np.nan,
        "tie_rate_diff_llm_minus_human": (
            float((llm == "Tie").mean() - (human == "Tie").mean()) if n else np.nan
        ),
    }
    if "pair_type" in group.columns:
        hum_llm = group[group["pair_type"] == "HUM-vs-LLM"]
        row["hum_vs_llm_n"] = len(hum_llm)
        row["human_prefers_llm_rate_in_hum_vs_llm"] = llm_preference_rate(hum_llm, "human_outcome")
        row["llm_prefers_llm_rate_in_hum_vs_llm"] = llm_preference_rate(hum_llm, "llm_outcome")
    for col in group_cols:
        row[col] = group[col].iloc[0]
    return row


def llm_preference_rate(df: pd.DataFrame, outcome_col: str) -> float:
    if df.empty:
        return np.nan
    prefers = []
    for _, row in df.iterrows():
        outcome = row[outcome_col]
        if outcome == "Tie":
            prefers.append(np.nan)
        elif outcome == "A_wins":
            prefers.append(hypothesis_type(row["Hypothesis_A_ID"]) == "LLM")
        elif outcome == "B_wins":
            prefers.append(hypothesis_type(row["Hypothesis_B_ID"]) == "LLM")
    valid = pd.Series(prefers, dtype="object").dropna()
    return float(valid.mean()) if len(valid) else np.nan


def score_for_hypothesis_rows(df: pd.DataFrame, outcome_col: str, source_label: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        if row[outcome_col] == "A_wins":
            a_score, b_score = 1.0, 0.0
        elif row[outcome_col] == "B_wins":
            a_score, b_score = 0.0, 1.0
        else:
            a_score, b_score = 0.5, 0.5
        rows.append(
            {
                "Dimension": row["Dimension"],
                "hypothesis_id": row["Hypothesis_A_ID"],
                "hypothesis_type": hypothesis_type(row["Hypothesis_A_ID"]),
                source_label: a_score,
            }
        )
        rows.append(
            {
                "Dimension": row["Dimension"],
                "hypothesis_id": row["Hypothesis_B_ID"],
                "hypothesis_type": hypothesis_type(row["Hypothesis_B_ID"]),
                source_label: b_score,
            }
        )
    return pd.DataFrame(rows)


def aggregate_hypothesis_scores(aligned: pd.DataFrame, model_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    human_scores = score_for_hypothesis_rows(aligned, "human_outcome", "human_score")
    llm_scores = score_for_hypothesis_rows(aligned, "llm_outcome", "llm_score")
    human_agg = (
        human_scores.groupby(["Dimension", "hypothesis_id", "hypothesis_type"], as_index=False)
        .agg(human_mean_score=("human_score", "mean"), human_n=("human_score", "size"))
    )
    llm_agg = (
        llm_scores.groupby(["Dimension", "hypothesis_id", "hypothesis_type"], as_index=False)
        .agg(llm_mean_score=("llm_score", "mean"), llm_n=("llm_score", "size"))
    )
    merged = human_agg.merge(llm_agg, on=["Dimension", "hypothesis_id", "hypothesis_type"], how="outer")
    merged["model"] = model_label
    corr_rows = []
    for keys, group in merged.groupby(["model", "Dimension"], dropna=False):
        model, dimension = keys
        valid = group[["human_mean_score", "llm_mean_score"]].dropna()
        corr_rows.append(
            {
                "model": model,
                "Dimension": dimension,
                "n_hypotheses": len(valid),
                "pearson_r": safe_corr(valid["human_mean_score"], valid["llm_mean_score"], "pearson"),
                "spearman_r": safe_corr(valid["human_mean_score"], valid["llm_mean_score"], "spearman"),
                "mean_score_diff_llm_minus_human": float(
                    (valid["llm_mean_score"] - valid["human_mean_score"]).mean()
                )
                if len(valid)
                else np.nan,
            }
        )
    return merged, pd.DataFrame(corr_rows)


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return np.nan
    if method == "spearman":
        left = left.rank(method="average")
        right = right.rank(method="average")
        method = "pearson"
    return float(left.corr(right, method=method))


def confusion_matrix(aligned: pd.DataFrame, model_label: str) -> pd.DataFrame:
    matrix = pd.crosstab(
        aligned["human_outcome"],
        aligned["llm_outcome"],
        rownames=["human_outcome"],
        colnames=["llm_outcome"],
        dropna=False,
    )
    matrix = matrix.reindex(index=OUTCOME_ORDER, columns=OUTCOME_ORDER, fill_value=0)
    matrix.insert(0, "model", model_label)
    return matrix.reset_index()


def align_runs(human_df: pd.DataFrame, llm_df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    human_dupes = human_df.duplicated(KEY_COLS).sum()
    llm_dupes = llm_df.duplicated(KEY_COLS).sum()
    if human_dupes or llm_dupes:
        raise ValueError(f"Duplicate keys found: human={human_dupes}, {model_label}={llm_dupes}")

    aligned = human_df.merge(
        llm_df,
        on=KEY_COLS,
        suffixes=("_human", "_llm"),
        how="outer",
        indicator=True,
    )
    missing = aligned[aligned["_merge"] != "both"]
    if not missing.empty:
        missing.to_csv(f"missing_alignment_{model_label}.csv", index=False)
        raise ValueError(f"{model_label} has {len(missing)} rows that do not align with human results")

    aligned = aligned.rename(
        columns={
            "Result_human": "human_result",
            "Result_llm": "llm_result",
            "outcome_human": "human_outcome",
            "outcome_llm": "llm_outcome",
            "pair_type_human": "pair_type",
        }
    )
    aligned["model"] = model_label
    aligned["exact_match"] = aligned["human_outcome"] == aligned["llm_outcome"]
    aligned["both_tie"] = (aligned["human_outcome"] == "Tie") & (aligned["llm_outcome"] == "Tie")
    aligned["human_tie_llm_not"] = (aligned["human_outcome"] == "Tie") & (aligned["llm_outcome"] != "Tie")
    aligned["llm_tie_human_not"] = (aligned["human_outcome"] != "Tie") & (aligned["llm_outcome"] == "Tie")
    return aligned


def build_summaries(aligned_frames: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    all_aligned = pd.concat(aligned_frames, ignore_index=True)
    summary_rows = []
    for model, group in all_aligned.groupby("model", sort=False):
        summary_rows.append(summarize_group(group, ["model"]))
    overall = pd.DataFrame(summary_rows)

    by_dimension = []
    for keys, group in all_aligned.groupby(["model", "Dimension"], sort=False):
        by_dimension.append(summarize_group(group, ["model", "Dimension"]))

    by_expert = []
    for keys, group in all_aligned.groupby(["model", "Expert_ID"], sort=False):
        by_expert.append(summarize_group(group, ["model", "Expert_ID"]))

    by_pair_type = []
    for keys, group in all_aligned.groupby(["model", "pair_type"], sort=False):
        by_pair_type.append(summarize_group(group, ["model", "pair_type"]))

    by_dimension_pair_type = []
    for keys, group in all_aligned.groupby(["model", "Dimension", "pair_type"], sort=False):
        by_dimension_pair_type.append(summarize_group(group, ["model", "Dimension", "pair_type"]))

    tie_behavior = (
        all_aligned.groupby(["model", "Dimension"], as_index=False)
        .agg(
            n=("exact_match", "size"),
            both_tie=("both_tie", "sum"),
            human_tie_llm_not=("human_tie_llm_not", "sum"),
            llm_tie_human_not=("llm_tie_human_not", "sum"),
            human_tie_rate=("human_outcome", lambda s: float((s == "Tie").mean())),
            llm_tie_rate=("llm_outcome", lambda s: float((s == "Tie").mean())),
        )
    )
    tie_behavior["both_tie_rate"] = tie_behavior["both_tie"] / tie_behavior["n"]

    confusion = pd.concat(
        [confusion_matrix(group, model) for model, group in all_aligned.groupby("model", sort=False)],
        ignore_index=True,
    )

    hypothesis_aggregates = []
    aggregate_correlations = []
    for model, group in all_aligned.groupby("model", sort=False):
        hyp, corr = aggregate_hypothesis_scores(group, model)
        hypothesis_aggregates.append(hyp)
        aggregate_correlations.append(corr)

    return {
        "overall": overall,
        "by_dimension": pd.DataFrame(by_dimension),
        "by_expert": pd.DataFrame(by_expert),
        "by_pair_type": pd.DataFrame(by_pair_type),
        "by_dimension_pair_type": pd.DataFrame(by_dimension_pair_type),
        "tie_behavior": tie_behavior,
        "confusion_matrix": confusion,
        "aggregate_hypothesis_scores": pd.concat(hypothesis_aggregates, ignore_index=True),
        "aggregate_score_correlations": pd.concat(aggregate_correlations, ignore_index=True),
        "aligned_rows": all_aligned,
    }


def write_outputs(output_dir: Path, summaries: dict[str, pd.DataFrame], metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "llm_human_consistency_analysis.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, df in summaries.items():
            safe_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

    for name, df in summaries.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_summary(output_dir / "summary.md", summaries)


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def write_markdown_summary(path: Path, summaries: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# LLM-Human Consistency Summary",
        "",
        "## Overall",
        "",
        "| model | n | exact | kappa | AC1 | AC2_quad | human tie | LLM tie |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summaries["overall"].iterrows():
        lines.append(
            "| {model} | {n} | {exact} | {kappa} | {ac1} | {ac2} | {htie} | {ltie} |".format(
                model=row["model"],
                n=int(row["n"]),
                exact=fmt(row["exact_agreement"]),
                kappa=fmt(row["cohen_kappa"]),
                ac1=fmt(row["gwet_ac1"]),
                ac2=fmt(row["gwet_ac2_quadratic"]),
                htie=fmt(row["human_tie_rate"]),
                ltie=fmt(row["llm_tie_rate"]),
            )
        )
    lines.extend(["", "## By Dimension", ""])
    lines.append("| model | dimension | n | exact | kappa | AC1 | human tie | LLM tie |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in summaries["by_dimension"].iterrows():
        lines.append(
            "| {model} | {dim} | {n} | {exact} | {kappa} | {ac1} | {htie} | {ltie} |".format(
                model=row["model"],
                dim=row["Dimension"],
                n=int(row["n"]),
                exact=fmt(row["exact_agreement"]),
                kappa=fmt(row["cohen_kappa"]),
                ac1=fmt(row["gwet_ac1"]),
                htie=fmt(row["human_tie_rate"]),
                ltie=fmt(row["llm_tie_rate"]),
            )
        )
    lines.extend(["", "## Aggregate Score Correlations", ""])
    lines.append("| model | dimension | hypotheses | pearson | spearman | mean diff |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in summaries["aggregate_score_correlations"].iterrows():
        lines.append(
            "| {model} | {dim} | {n} | {pearson} | {spearman} | {diff} |".format(
                model=row["model"],
                dim=row["Dimension"],
                n=int(row["n_hypotheses"]),
                pearson=fmt(row["pearson_r"]),
                spearman=fmt(row["spearman_r"]),
                diff=fmt(row["mean_score_diff_llm_minus_human"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.llm_run:
        llm_runs = [JudgeRun(label, Path(path)) for label, path in args.llm_run]
    else:
        llm_runs = default_llm_runs()
    if not llm_runs:
        raise ValueError("No LLM runs were provided and no default run files were found.")

    human_df = read_results(args.human_results, "human")
    aligned_frames = []
    for run in llm_runs:
        llm_df = read_results(run.path, run.label)
        aligned_frames.append(align_runs(human_df, llm_df, run.label))

    summaries = build_summaries(aligned_frames)
    metadata = {
        "human_results": str(args.human_results),
        "llm_runs": [{"label": run.label, "path": str(run.path)} for run in llm_runs],
        "key_columns": KEY_COLS,
        "outcomes": OUTCOME_ORDER,
        "rope": list(args.rope),
        "notes": [
            "Agreement metrics are computed on normalized outcomes: A_wins, B_wins, Tie.",
            "Gwet AC2 uses quadratic weights with Tie treated as the middle category.",
            "Hypothesis aggregate scores use win=1, tie=0.5, loss=0.",
        ],
    }
    write_outputs(args.output_dir, summaries, metadata)
    print(f"Wrote analysis workbook and CSV files to: {args.output_dir}")
    print(summaries["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
