# LLM-as-Judge Expert Simulation and Evaluation

## Purpose

This project simulates expert pairwise hypothesis evaluation with large language
models, compares LLM judgments with human expert ratings, and estimates
source-level effects with a Bayesian Davidson–Bradley–Terry–Luce model.

The shared folder includes experiment and analysis code, input workbooks,
completed LLM and human comparison results, Bayesian posterior files,
computation logs, summary tables, and publication-ready figures.

## Project Structure

```text
LLMJudge/
├── data/
│   ├── All_Assignment_Pairs.xlsx
│   └── All_Comparison_Results.xlsx
├── LLMJudge_results/
│   ├── comparison_results_claude.xlsx
│   ├── comparison_results_gemini.xlsx
│   └── comparison_results_gpt.xlsx
├── analysis_results/
│   └── llm_human_consistency/
│       ├── figure.py
│       ├── figure/
│       │   ├── llm_human_consistency.png
│       │   └── llm_human_consistency.pdf
│       └── analysis tables and summaries
├── LLM_as_Judge_BTL_computation/
│   ├── run_bayesian_btl_analysis.py
│   ├── submit_btl_full.slurm
│   ├── submit_btl_reuse_human.slurm
│   ├── submit_btl_smoke.slurm
│   └── HPC_run_results/
│       ├── logs/
│       └── bayesian_btl_results/
│           ├── model_based_comparison.png
│           ├── model_based_comparison.pdf
│           ├── model_based_comparison_values.csv
│           ├── LLM_judge_bayesian_calculations.py
│           ├── LLM_judge_bayesian_calculations_data.json
│           ├── LLM_judge_bayesian_calculations.xlsx
│           ├── fitted NetCDF posterior files
│           └── summary tables and diagnostic plots
├── run_llm_expert_survey.py
├── smoke_test_failed_pairs.py
├── analyze_llm_human_consistency.py
├── expertrankhypothesisv2.py
├── run_bayesian_btl_analysis.py
├── submit_btl.slurm
├── submit_btl_smoke.slurm
├── requirements.txt
└── environment_btl.yml
```

## Environment Installation

Python 3.11 is recommended.

Using `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, create the Bayesian environment with Conda:

```powershell
conda env create -f environment_btl.yml
conda activate llmjudge-btl
```

## Running the Project

Run the following commands from the `LLMJudge` directory unless noted
otherwise.

### Check an LLM experiment without API calls

```powershell
python run_llm_expert_survey.py --dry-run
```

The default dry run plans 1,350 model calls and 4,050 comparison rows.

### Run the LLM expert survey

Store the API key in an environment variable:

```powershell
$env:EXPERT_API_KEY = "YOUR_API_KEY"
python run_llm_expert_survey.py
```

To use a different OpenAI-compatible endpoint or model:

```powershell
$env:EXPERT_API_KEY = "YOUR_API_KEY"
$env:EXPERT_BASE_URL = "YOUR_BASE_URL"
python run_llm_expert_survey.py --model-name "MODEL_NAME"
```

A small pilot run can be started with:

```powershell
python run_llm_expert_survey.py --expert-ids E01 --max-pairs 3 --run-label pilot
```

LLM outputs are written under `LLMJudge_results/`.

### Analyze agreement with human experts

```powershell
python analyze_llm_human_consistency.py
python analysis_results/llm_human_consistency/figure.py
```

The four panels are ordered as source preference, overall agreement, agreement
by dimension, and tie behavior.

### Run the Bayesian BTL analysis

```powershell
python run_bayesian_btl_analysis.py
```

The driver uses `All_Comparison_Results.xlsx` and the three comparison
workbooks under `LLMJudge_results/` by default. This is a computationally
intensive MCMC analysis. Parameters such as `--draws`, `--tune`, `--chains`,
`--cores`, and `--target-accept` should be changed only when a different
analysis is intended.

### Regenerate the Bayesian comparison figure

```powershell
Set-Location LLM_as_Judge_BTL_computation/HPC_run_results/bayesian_btl_results
python plot_bayesian_btl_comparison.py
```

This command reads the retained NetCDF posterior files and regenerates
`model_based_comparison.png`, `model_based_comparison.pdf`, and
`model_based_comparison_values.csv`.

### Extract numerical summaries from the retained posterior files

```powershell
Set-Location LLM_as_Judge_BTL_computation/HPC_run_results/bayesian_btl_results
python LLM_judge_bayesian_calculations.py
```

The script reads all 12 retained NetCDF posterior files without refitting or
modifying them. It writes the extracted posterior summaries to
`LLM_judge_bayesian_calculations_data.json`. The accompanying
`LLM_judge_bayesian_calculations.xlsx` organizes the same results into
review-ready worksheets for source effects, direct LLM-minus-human
comparisons, tie probabilities, manuscript ranges, calculation definitions,
and the input-file inventory.

### Submit an HPC job

The Slurm scripts are retained. Personal absolute paths have been replaced with
portable variables:

- `SLURM_SUBMIT_DIR` selects the submitted project directory.
- `HPC_CONDA_ROOT` can override the Conda installation location.
- If `HPC_CONDA_ROOT` is not set, `$HOME/WORK/miniconda3` is used.

Example:

```bash
export HPC_CONDA_ROOT="$HOME/WORK/miniconda3"
sbatch submit_btl_smoke.slurm
```

## Inputs

- `data/All_Assignment_Pairs.xlsx`: expert assignments, hypothesis pairs, and
  hypothesis text for the LLM survey.
- `All_Comparison_Results.xlsx` and
  `data/All_Comparison_Results.xlsx`: human expert pairwise judgments.
- `LLMJudge_results/comparison_results_*.xlsx`: completed LLM pairwise
  judgments.
- `LLM_as_Judge_BTL_computation/HPC_run_results/bayesian_btl_results/*/*.nc`:
  retained fitted posterior samples used by the Bayesian comparison figure.

## Outputs

Each LLM run writes a model-specific directory under `LLMJudge_results/` with:

- `comparison_results.xlsx`: normalized pairwise judgments.
- `raw_model_calls.jsonl`: prompts, parsed answers, raw messages, and API
  responses.
- `failed_model_calls.jsonl`: failed calls, when present.
- `run_metadata.json`: run settings and completion counts.

The consistency analysis writes summary tables, a workbook, metadata, and
`llm_human_consistency.png`/`.pdf` under
`analysis_results/llm_human_consistency/`.

The Bayesian computation folder retains all completed NetCDF posterior files,
CSV/XLSX summaries, diagnostic images, logs, and the final model-based
comparison figure.

## Figure Palette

Both final figures use the requested palette:

- Human: `#999999`
- Claude: `#015493`
- Gemini: `#019092`
- GPT: `#F4A99B`

The Bayesian figure also uses distinct marker shapes so interpretation does not
depend on color alone.

## Security Notes

- Keep `EXPERT_API_KEY` in the environment and never commit it.
- Review newly generated `raw_model_calls.jsonl` before sharing because it
  contains full prompts and model responses.
- Slurm files are retained because job scripts are not credentials. Personal
  absolute paths have been replaced with portable environment variables.
- Existing computation results and logs are retained. The single personal
  absolute path found in a preserved error log has been anonymized without
  deleting the error message.
