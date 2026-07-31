# Bayesian Hierarchical Davidson–BTL Analysis of Expert Ratings

## Project purpose

This project evaluates whether experts favor LLM-generated or human-proposed
urban-biodiversity hypotheses. It fits a hierarchical Davidson extension of the
Bradley–Terry–Luce (BTL) model separately for novelty, significance, and
testability. The model estimates:

- the population-level LLM source effect (`mu_LLM`);
- between-expert heterogeneity in that effect (`sigma_LLM`);
- expert-specific LLM effects;
- the probability of a tied judgment.

The project also provides convergence diagnostics, expert-level forest plots,
and publication-ready Bayesian summary figures.

## File structure

```text
Experts’s ratings(BTL Model)/
├── expertrankhypothesis.py
├── README.md
├── requirements.txt
├── figure/
│   ├── Bayesian_summary.py
│   ├── Bayesian_summary.png
│   ├── Bayesian_summary.pdf
│   └── Bayesian_summary_values.csv
└── ResultsFinal/
    ├── Checktraceplots.py
    ├── Bayesian_summary.py
    ├── Bayesian_summary.png
    ├── Bayesian_summary.pdf
    ├── Bayesian_summary_values.csv
    ├── idata_<Dimension>.nc
    ├── summary_<Dimension>.csv
    └── expert_bias_forest_<Dimension>.png
```

- `expertrankhypothesis.py` loads the comparison data, fits the hierarchical
  Davidson–BTL model, and writes new model outputs to `Results/`.
- `ResultsFinal/` contains the final saved posterior samples, parameter
  summaries, diagnostic plots, and a self-contained Bayesian summary plotting
  script.
- `figure/Bayesian_summary.py` is the publication-ready plotting script. It
  reads the final NetCDF files from `ResultsFinal/` and writes its outputs to
  `figure/`.
- `ResultsFinal/Checktraceplots.py` displays trace and rank plots for the
  novelty model.

## Environment installation

Python 3.9 or later is recommended. From the project directory, create and
activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Input data

Model fitting expects the following workbook in the project root:

```text
All_Comparison_Results.xlsx
```

The workbook must contain these columns:

| Column | Description |
| --- | --- |
| `Expert_ID` | Unique expert identifier |
| `Hypothesis_A_ID` | Identifier of the first hypothesis |
| `Hypothesis_B_ID` | Identifier of the second hypothesis |
| `Dimension` | Evaluation dimension |
| `Result` | `Tie` or the identifier of the selected hypothesis |

Hypothesis identifiers are expected to use the `LLM_` and `HUM_` prefixes. The
provided summary plotting scripts expect posterior files for `Novelty`,
`Significance`, and `Testability`.

## Running the project

### Fit the Bayesian model

```powershell
python expertrankhypothesis.py
```

Model fitting uses multiple MCMC chains and can take substantial time. New
posterior samples, summaries, and expert-level forest plots are written to
`Results/`.

### Generate the publication-ready Bayesian summary

The supplied final posterior files are already stored in `ResultsFinal/`.
Generate the summary figure and its accompanying values with:

```powershell
python figure/Bayesian_summary.py
```

This creates:

- `figure/Bayesian_summary.png`;
- `figure/Bayesian_summary.pdf`;
- `figure/Bayesian_summary_values.csv`.

An alternative self-contained layout can be generated from the same posterior
files with:

```powershell
python ResultsFinal/Bayesian_summary.py
```

Its outputs are written directly to `ResultsFinal/`.

### Inspect sampling diagnostics

```powershell
python ResultsFinal/Checktraceplots.py
```

This command opens interactive trace and rank plots for the novelty model.

## Outputs

- `idata_<Dimension>.nc`: complete ArviZ posterior data for each dimension;
- `summary_<Dimension>.csv`: posterior parameter summaries and convergence
  diagnostics;
- `expert_bias_forest_<Dimension>.png`: expert-specific LLM-effect forest
  plots;
- `Bayesian_summary.png` and `Bayesian_summary.pdf`: three-panel posterior
  summary figures;
- `Bayesian_summary_values.csv`: posterior means and 95% highest-density
  intervals used in the summary figure.

