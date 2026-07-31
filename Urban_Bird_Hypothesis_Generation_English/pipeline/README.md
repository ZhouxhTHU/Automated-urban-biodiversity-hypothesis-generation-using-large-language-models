# Pipeline Guide

## Execution order

Run the following commands in this directory:

```bash
python phase1.py
python phase2.py
```

Do not run these files directly from the project root because their input and checkpoint paths are resolved relative to the `pipeline/` working directory.

## Main files

- `phase1.py` generates, screens, and evaluates initial hypotheses from the literature corpus and retained checkpoints.
- `phase2.py` reads Phase 1 hypotheses and the inspiration pool, retrieves supplementary literature, and refines the hypotheses.
- `agent.py` defines the expert, leading-expert, screener, and critic agents.
- `prompt.py` stores prompts used by both phases.
- `utils.py` provides JSON loading, API retry, format repair, token tracking, and clustering-evaluation utilities.
- `chain_of_papers.py` retrieves supplementary literature, builds paper chains, and assigns papers to experts.
- `config.py` stores data paths, model configuration, the research topic, and workflow parameters.
- `log.py` creates run-log directories and saves configuration and prompt snapshots.
- `45Hypo/` contains only the three checkpoint inputs directly referenced by the current entry points.

## Inputs and outputs

The Phase 1 corpus input is:

```text
../Corpus_of_species_interaction/bird1487/UrbanBirdDiversity_Corpus_1487.json
```

The current Phase 1 configuration also uses `all_inspirations.json` and `inspirations_pool.json` under `45Hypo/`. Phase 2 uses the same inspiration pool and the `initial_hypothesis.json` in the selected checkpoint directory.

Logs, configuration summaries, prompt snapshots, token statistics, checkpoints, and final results are written to timestamped directories created by the code. Historical instances of those generated files are not included in this cleaned repository, except for the three JSON inputs required by the current entry points.

## Dependency installation

```bash
pip install -r requirements.txt
```

The dependency list is derived from third-party imports in the retained files.
