# Descriptive Analysis of Expert Hypothesis Ratings

## Project purpose

This project processes expert pairwise evaluations of human-proposed and
LLM-generated urban-biodiversity hypotheses. It converts the survey workbooks
into a tidy comparison table, summarizes selection outcomes at the hypothesis,
expert, and evaluation-dimension levels, and creates publication-ready
descriptive figures.

The three evaluation dimensions are:

- novelty;
- significance;
- testability.

## File structure

```text
Experts’s ratings(DescriptionAnalysis)/
├── All_Assignment_Pairs.xlsx
├── All_Comparison_Results.xlsx
├── Hypothesis_Analysis_Result.xlsx
├── Convert_to_TrainingData.py
├── Statistical_analysis.py
├── README.md
├── requirements.txt
├── Expert_Results/
│   └── E<id> Urban Biodiversity Hypothesis Project.xlsx
└── figure/
    ├── figure_main.py
    ├── combine_figures.py
    └── figures/
        ├── figure1_cross_source_counts.png
        ├── figure2_expert_llm_share_by_dimension_boxplot.png
        ├── figure3_expert_tie_rate_by_dimension_boxplot.png
        ├── expert_dimension_plot_data.xlsx
        ├── figureX_descriptive_outcomes_cross_source_comparisons.png
        ├── figureX_descriptive_outcomes_cross_source_comparisons.pdf
        └── figureX_descriptive_outcomes_cross_source_comparisons_caption.txt
```

- `Convert_to_TrainingData.py` combines the assignment table and expert survey
  responses into `All_Comparison_Results.xlsx`.
- `Statistical_analysis.py` calculates descriptive results and writes
  `Hypothesis_Analysis_Result.xlsx`.
- `figure/figure_main.py` is the primary plotting entry point. It now generates
  the three individual panels, the expert-by-dimension plotting data, and the
  combined a/b/c figure in one run.
- `figure/combine_figures.py` is retained for compatibility, but it is no longer
  required when `figure_main.py` is used.

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

## Input files

### Assignment table

`All_Assignment_Pairs.xlsx` maps each expert and pair number to the two
hypotheses shown in that comparison. The scripts use the following fields:

- `Expert_ID`;
- `Pair_Number`;
- `Hypothesis_A_ID`;
- `Hypothesis_B_ID`.

### Expert survey workbooks

`Expert_Results/` contains one workbook per expert. The filename must begin with
the expert identifier, and the second data row contains the actual responses.
Question columns begin with `Which hypothesis`.

### Comparison table

`All_Comparison_Results.xlsx` contains the normalized pairwise results used for
downstream analysis:

- `Expert_ID`;
- `Hypothesis_A_ID`;
- `Hypothesis_B_ID`;
- `Dimension`;
- `Result`.

## Running the project

Run the data-conversion and statistical scripts from the project directory:

```powershell
python Convert_to_TrainingData.py
python Statistical_analysis.py
```

Generate all individual and combined figures from the `figure/` directory:

```powershell
cd figure
python figure_main.py
```

No separate `combine_figures.py` command is required.

## Figure colors

The plots use a color-blind-friendly palette recommended by the
[Nature Research Figure Guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/):

| Use | Color |
| --- | --- |
| Human outcomes / novelty | Sky blue `#56B4E9` |
| LLM outcomes / significance | Orange `#E69F00` |
| Ties / testability | Reddish purple `#CC79A7` |

Black outlines are used to maintain contrast on a white background. The
sky-blue, orange, and reddish-purple combination provides a balanced contrast
while remaining color-blind-friendly.

## Outputs

`Statistical_analysis.py` creates `Hypothesis_Analysis_Result.xlsx`, including:

- overall selection counts;
- cross-source human-versus-LLM results;
- within-source comparisons;
- hypothesis-level summaries;
- hypothesis-level summaries by dimension;
- expert-level summaries;
- expert-level direct cross-source summaries.

`figure_main.py` creates:

- three standalone PNG panels;
- `expert_dimension_plot_data.xlsx`;
- a combined PNG figure;
- a combined PDF figure;
- a plain-text figure caption.
