# Expert Open-Ended Feedback Analysis

## Project purpose

This project analyzes experts' open-ended comments about LLM-generated urban
biodiversity hypotheses. The analysis pipeline uses an OpenAI-compatible model
to convert free-text responses into structured qualitative codes, including:

- sentiment category and score;
- primary topics;
- cited advantages (`pros`);
- cited limitations (`cons`).

Similar descriptions are grouped with fuzzy matching. The project then produces
sentiment, pros, and cons summaries suitable for review and visualization.

## File structure

```text
Expertoptions/
├── Expertopinions.csv
├── Opinionanalysis.py
├── README.md
├── requirements.txt
└── Outputs/
    ├── coded_responses_empirical.csv
    ├── sentiment_distribution.png
    ├── pros_cons_frequencies.png
    ├── figure.py
    └── Outputs/
        ├── open_feedback_abc.png
        ├── open_feedback_abc.pdf
        └── open_feedback_abc_counts.csv
```

- `Expertopinions.csv` contains the original expert identifiers and open-ended
  feedback.
- `Opinionanalysis.py` performs API-assisted qualitative coding, fuzzy grouping,
  and the initial visualizations.
- `Outputs/coded_responses_empirical.csv` stores the structured coding results
  and allows figures to be regenerated without another API call.
- `Outputs/figure.py` creates the combined a/b/c summary figure from the existing
  coded CSV.

## Environment installation

Python 3.9 or later is recommended. From the project directory, create and
activate a virtual environment and install the dependencies:

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

The default input is:

```text
Expertopinions.csv
```

It must contain the following columns:

| Column | Description |
| --- | --- |
| `Expert_ID` | Unique expert identifier |
| `Feedback` | Expert's open-ended response |

`Opinionanalysis.py` can also read Excel workbooks when its input configuration
is updated.

## Running the full qualitative analysis

Before running the complete pipeline, review `Expert_MODEL_CONFIG` in
`Opinionanalysis.py` and provide valid credentials, model information, and the
correct OpenAI-compatible API endpoint for your provider.

Do not commit or share live API credentials. The full pipeline sends the expert
feedback to the configured external provider and may incur usage charges.

Run the analysis from the project directory:

```powershell
python Opinionanalysis.py
```

The script loads `Expertopinions.csv`, requests structured qualitative coding,
groups similar pros and cons, and writes the following files to `Outputs/`:

- `coded_responses_empirical.csv`;
- `sentiment_distribution.png`;
- `pros_cons_frequencies.png`.

## Regenerating the combined figure without API calls

The combined figure can be reproduced from the existing coded CSV without
contacting an external model:

```powershell
cd Outputs
python figure.py
```

The script reads `coded_responses_empirical.csv` and writes:

- `Outputs/open_feedback_abc.png`;
- `Outputs/open_feedback_abc.pdf`;
- `Outputs/open_feedback_abc_counts.csv`.

These paths are relative to the `Expertoptions/Outputs/` working directory, so
the generated files are located in `Expertoptions/Outputs/Outputs/`.

## Figure colors

The combined figure uses the following fixed colors:

| Panel | Category | Color |
| --- | --- | --- |
| a | Positive | `#009E73` |
| a | Mixed | `#E69F00` |
| a | Neutral | `#7F7F7F` |
| a | Negative | `#D55E00` |
| b | Pros | `#0072B2` |
| c | Cons | `#D55E00` |

## Output data

`coded_responses_empirical.csv` includes the source feedback plus the structured
sentiment, topic, pros, and cons fields. `open_feedback_abc_counts.csv` contains
the exact category and descriptor frequencies plotted in panels a, b, and c.

