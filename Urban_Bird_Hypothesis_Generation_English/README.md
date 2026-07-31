# Urban Bird Diversity Scientific Hypothesis Generation

## Project overview

This project uses a multi-agent, large-language-model workflow to extract research inspirations from the urban bird diversity literature, generate initial scientific hypotheses, and further refine those hypotheses through supplementary literature retrieval, expert discussion, and critique iterations.

The workflow has two main phases:

1. `phase1.py` reads the literature corpus, extracts and screens inspirations, and generates and evaluates initial hypotheses.
2. `phase2.py` reads the Phase 1 hypotheses and inspiration pool, retrieves supplementary literature, organizes expert discussions, and produces refined hypothesis results.

The retained dataset is the urban bird diversity literature corpus supplied with the original project. The available files do not document a more specific collection platform, search strategy, or license, so data provenance and permissions should be confirmed separately before public distribution.

## Directory structure

```text
.
├── pipeline/
│   ├── 45Hypo/                         # Checkpoint inputs referenced by the current code
│   ├── agent.py                        # Agent definitions and creation
│   ├── chain_of_papers.py              # Supplementary retrieval and paper-chain construction
│   ├── config.py                       # Model, path, and experiment parameters
│   ├── log.py                          # Logging and configuration-summary utilities
│   ├── phase1.py                       # Phase 1 entry point
│   ├── phase2.py                       # Phase 2 entry point
│   ├── prompt.py                       # Prompt definitions
│   ├── utils.py                        # Data-processing and shared utilities
│   ├── README.md                       # Detailed pipeline notes
│   └── requirements.txt
├── Corpus_of_species_interaction/
│   └── bird1487/
│       └── UrbanBirdDiversity_Corpus_1487.json
├── .gitignore
├── README.md
└── requirements.txt
```

The data directory remains named `Corpus_of_species_interaction/` instead of `corpus/` because `pipeline/config.py` uses that relative path. Keeping the name preserves the original input-path logic exactly.

## Environment installation

The original project does not specify a Python version or dependency versions. Install the third-party packages statically imported by the retained code in an isolated virtual environment:

```bash
python -m venv .venv
```

After activating the environment, run:

```bash
pip install -r requirements.txt
```

The root and `pipeline/` copies of `requirements.txt` are identical; either may be used according to the current working directory.

## Running instructions

The code resolves paths relative to `pipeline/`, so enter that directory first:

```bash
cd pipeline
```

Run Phase 1:

```bash
python phase1.py
```

The current `phase1.py` reads:

- `../Corpus_of_species_interaction/bird1487/UrbanBirdDiversity_Corpus_1487.json`
- `45Hypo/.../all_inspirations.json`
- `45Hypo/.../inspirations_pool.json`

The retained checkpoints cause the current configuration to continue from existing inspiration data. New results, configuration summaries, prompt snapshots, logs, and token statistics are written to timestamped result directories created by the code.

Run Phase 2:

```bash
python phase2.py
```

The current `phase2.py` reads the hard-coded `initial_hypothesis.json` and `inspirations_pool.json` under `45Hypo/`. It writes refined results to a new timestamped `phase2/` directory under the corresponding hypothesis directory; the primary output is `final_results.json`.

Before running, review the model services, API access, and parameters in `pipeline/config.py` for compatibility with the target environment. Some features access large-language-model services and literature services such as Semantic Scholar.

## Dataset description

- File: `Corpus_of_species_interaction/bird1487/UrbanBirdDiversity_Corpus_1487.json`
- Records: 1,487 literature items
- Format: UTF-8 JSON array
- Fields: `title`, `abstract`, `year`, and `ID`
- Contents: titles, abstracts, years, and project-local identifiers for literature concerning urban bird diversity
- Provenance note: the data file was supplied with the original project; no more specific source or license statement was found in the available project files

## Security note before sharing

The original `config.py` contains API credentials and service endpoints. In accordance with the task constraints, this cleanup did not modify any credential, parameter, or API call. Before publishing the repository or sharing it with an untrusted party, the credential owner should rotate or revoke the credentials and decide how to configure secrets safely for the intended audience.

## Cleanup scope

This repository retains only the transitive dependencies of the main workflow, the three checkpoint JSON files directly referenced by the current code, and the 1,487-item literature corpus actually used by the pipeline. Historical logs, caches, run snapshots, intermediate outputs, old copies, and auxiliary experiment files not referenced by the main workflow were not copied.
