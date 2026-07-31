"""
Run Bayesian Davidson-BTL analyses for human and LLM-as-Judge results.

This is a driver around expertrankhypothesisv2.py. It does not modify the
original model code; it only sets runtime options, points the model at each
dataset, and summarizes posterior differences between LLM judges and humans.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

import expertrankhypothesisv2 as btl


DIMENSIONS = ["Novelty", "Significance", "Testability"]


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTL posterior analyses for human and LLM judges.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=4000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--rope", type=float, nargs=2, default=(-0.1, 0.1), metavar=("LOW", "HIGH"))
    parser.add_argument("--output-dir", type=Path, default=Path("bayesian_btl_results"))
    parser.add_argument("--human", type=Path, default=Path("All_Comparison_Results.xlsx"))
    parser.add_argument(
        "--human-idata-dir",
        type=Path,
        default=None,
        help="Directory containing existing human idata_<Dimension>.nc files. If set, human is not resampled.",
    )
    parser.add_argument("--claude", type=Path, default=Path("claude_comparison_results.xlsx"))
    parser.add_argument("--gemini", type=Path, default=Path("gemini_comparison_results.xlsx"))
    parser.add_argument("--gpt", type=Path, default=Path("gpt_comparison_results.xlsx"))
    parser.add_argument("--dimensions", nargs="+", default=DIMENSIONS)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def posterior_values(idata: az.InferenceData, variable: str) -> np.ndarray:
    return idata.posterior[variable].values.reshape(-1)


def summarize_posterior(idata: az.InferenceData, dataset: str, dimension: str, rope: tuple[float, float]) -> dict:
    mu = posterior_values(idata, "mu_LLM")
    sigma = posterior_values(idata, "sigma_LLM")
    tie = posterior_values(idata, "tie_logit")
    return {
        "dataset": dataset,
        "dimension": dimension,
        "mu_LLM_mean": float(mu.mean()),
        "mu_LLM_q2.5": float(np.quantile(mu, 0.025)),
        "mu_LLM_q97.5": float(np.quantile(mu, 0.975)),
        "P_mu_gt_0": float((mu > 0).mean()),
        "P_mu_lt_0": float((mu < 0).mean()),
        "P_mu_in_ROPE": float(((mu >= rope[0]) & (mu <= rope[1])).mean()),
        "sigma_LLM_mean": float(sigma.mean()),
        "sigma_LLM_q2.5": float(np.quantile(sigma, 0.025)),
        "sigma_LLM_q97.5": float(np.quantile(sigma, 0.975)),
        "tie_logit_mean": float(tie.mean()),
        "tie_logit_q2.5": float(np.quantile(tie, 0.025)),
        "tie_logit_q97.5": float(np.quantile(tie, 0.975)),
    }


def compare_to_human(
    idata_by_key: dict[tuple[str, str], az.InferenceData],
    dataset: str,
    dimension: str,
    rope: tuple[float, float],
) -> dict:
    human_mu = posterior_values(idata_by_key[("human", dimension)], "mu_LLM")
    judge_mu = posterior_values(idata_by_key[(dataset, dimension)], "mu_LLM")
    n = min(len(human_mu), len(judge_mu))
    delta = judge_mu[:n] - human_mu[:n]
    return {
        "dataset": dataset,
        "dimension": dimension,
        "delta_mu_mean": float(delta.mean()),
        "delta_mu_q2.5": float(np.quantile(delta, 0.025)),
        "delta_mu_q97.5": float(np.quantile(delta, 0.975)),
        "P_delta_mu_gt_0": float((delta > 0).mean()),
        "P_delta_mu_lt_0": float((delta < 0).mean()),
        "P_delta_mu_in_ROPE": float(((delta >= rope[0]) & (delta <= rope[1])).mean()),
    }


def run_one_dataset(
    spec: DatasetSpec,
    dimension: str,
    output_dir: Path,
    skip_existing: bool,
) -> az.InferenceData:
    dataset_dir = output_dir / spec.label
    dataset_dir.mkdir(parents=True, exist_ok=True)
    nc_path = dataset_dir / f"idata_{dimension}.nc"
    if skip_existing and nc_path.exists():
        print(f"[skip] {spec.label} / {dimension}: {nc_path}")
        return az.from_netcdf(nc_path)

    btl.OUTPUT_DIR = dataset_dir
    df = btl.load_and_preprocess_data(spec.path)
    _, idata, experts = btl.build_and_run_model(df, dimension)
    btl.save_diagnostics(idata, dimension)
    try:
        btl.plot_expert_bias(idata, experts, dimension)
    except Exception as exc:
        print(f"[warn] Could not create forest plot for {spec.label} / {dimension}: {exc}")
    return idata


def load_existing_human_idata(human_idata_dir: Path, dimension: str) -> az.InferenceData:
    nc_path = human_idata_dir / f"idata_{dimension}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing existing human posterior file: {nc_path}")
    print(f"[load] human / {dimension}: {nc_path}")
    return az.from_netcdf(nc_path)


def write_manifest(args: argparse.Namespace, datasets: list[DatasetSpec]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "rope": args.rope,
        "datasets": [{"label": spec.label, "path": str(spec.path)} for spec in datasets],
        "dimensions": args.dimensions,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    btl.DRAWS = args.draws
    btl.TUNE = args.tune
    btl.CHAINS = args.chains
    btl.CORES = args.cores
    btl.TARGET_ACCEPT = args.target_accept

    datasets = [
        DatasetSpec("human", args.human),
        DatasetSpec("claude", args.claude),
        DatasetSpec("gemini-3.5-flash-thinking", args.gemini),
        DatasetSpec("gpt-5.5", args.gpt),
    ]
    missing = [
        str(spec.path)
        for spec in datasets
        if not (spec.label == "human" and args.human_idata_dir) and not spec.path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, datasets)

    idata_by_key: dict[tuple[str, str], az.InferenceData] = {}
    posterior_rows = []
    rope = (float(args.rope[0]), float(args.rope[1]))

    for spec in datasets:
        for dimension in args.dimensions:
            print(f"\n=== Running {spec.label} / {dimension} ===", flush=True)
            if spec.label == "human" and args.human_idata_dir:
                idata = load_existing_human_idata(args.human_idata_dir, dimension)
            else:
                idata = run_one_dataset(spec, dimension, args.output_dir, args.skip_existing)
            idata_by_key[(spec.label, dimension)] = idata
            posterior_rows.append(summarize_posterior(idata, spec.label, dimension, rope))

    comparison_rows = []
    for spec in datasets:
        if spec.label == "human":
            continue
        for dimension in args.dimensions:
            comparison_rows.append(compare_to_human(idata_by_key, spec.label, dimension, rope))

    posterior_df = pd.DataFrame(posterior_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    posterior_df.to_csv(args.output_dir / "posterior_summary.csv", index=False)
    comparison_df.to_csv(args.output_dir / "delta_mu_vs_human.csv", index=False)

    workbook_path = args.output_dir / "bayesian_btl_posterior_comparison.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        posterior_df.to_excel(writer, sheet_name="posterior_summary", index=False)
        comparison_df.to_excel(writer, sheet_name="delta_mu_vs_human", index=False)

    latest_dir = Path("Results")
    if latest_dir.exists() and latest_dir.is_dir():
        shutil.copytree(latest_dir, args.output_dir / "legacy_results_copy", dirs_exist_ok=True)

    print("\nDone.")
    print(f"Posterior summary: {args.output_dir / 'posterior_summary.csv'}")
    print(f"Delta comparison:  {args.output_dir / 'delta_mu_vs_human.csv'}")
    print(f"Workbook:          {workbook_path}")


if __name__ == "__main__":
    main()
