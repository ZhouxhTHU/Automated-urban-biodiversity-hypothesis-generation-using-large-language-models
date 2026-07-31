# Clustering 1,487 Papers and Selecting the Top Five per Cluster

This folder contains only the data and code directly needed to reproduce the
following workflow:

1. Use the curated library of 1,487 papers as the input dataset.
2. Concatenate each paper's title and abstract. Four source records do not have
   abstracts and therefore use the title alone.
3. Generate 3,072-dimensional vector representations with OpenAI's
   `text-embedding-3-large` model.
4. Cluster the embeddings into nine semantic groups using K-medoids with PAM,
   cosine distance, and `random_state = 42`.
5. Calculate the annual citation rate using 2026 as the fixed reference year:

   `TCperYear = Times Cited, WoS Core / (2026 - txt_year + 1)`

6. Rank papers by `TCperYear` within each cluster and retain the top five,
   producing a final set of 45 representative papers.

## Folder Structure

- `01_Input_Data/1487_Papers_Titles_Abstracts_and_Citations.xlsx`
  - Final clustering input with 1,487 records.
  - Includes titles, publication years, abstracts, and Web of Science citation
    counts.
- `02_Embedding_Cache/embeddings_text-embedding-3-large_1487x3072.npy`
  - Archived embedding matrix aligned to the input row order.
  - Shape: `1,487 × 3,072`.
- `03_Clustering/01_KMedoids_9_Clusters.py`
  - Reproducible clustering script.
  - Reads the OpenAI API key from the `OPENAI_API_KEY` environment variable and
    contains no plaintext key.
- `03_Clustering/1487_Papers_KMedoids_9_Clusters.xlsx`
  - Archived nine-cluster assignment for all 1,487 papers.
- `04_Top5_Selection/02_Calculate_Annual_Citations_and_Select_Top5_per_Cluster.R`
  - Reproducible annual-citation calculation and within-cluster selection script.
- `04_Top5_Selection/1487_Papers_9_Clusters_and_Annual_Citations_2026.xlsx`
  - Full dataset with cluster assignments and annual citation rates.
- `04_Top5_Selection/Final_45_Papers_Top5_per_Cluster.csv`
  - Final 45 representative papers, with exactly five papers from each cluster.
- `05_Final_45_PDFs/`
  - PDF files for the 45 representative papers.

## Software Requirements

The Python script requires:

- `pandas`
- `numpy`
- `openai`
- `scikit-learn`
- `scikit-learn-extra`
- `openpyxl`

The R script requires:

- `dplyr`
- `readxl`
- `writexl`

Both scripts write reproduced results to new filenames containing
`Reproduced`; they do not overwrite the archived outputs.

## Data Preservation Note

The spreadsheet and CSV contents were copied without modification so that the
archived embedding cache, cluster assignments, and selected papers remain
exactly reproducible. One non-selected source record (row 1,478 in the Excel
files) contains Chinese character fragments in its abstract, apparently from
an earlier text-processing or encoding issue. It was intentionally preserved;
editing it would require regenerating the embeddings and rerunning the
clustering analysis.

## Scope

Historical batch exports, spreadsheet-merging files, citation-matching
intermediates, unmatched-record files, and author/expert overlap analyses were
intentionally excluded from this organized version.

The user-provided source folder has not been modified or deleted.

## Result Not Present in the Source Folder

No standalone table containing the 45 core scientific hypotheses extracted by
an LLM, and no file identifiable as `Table SX`, was present in the source
folder. This package therefore preserves the selected paper list, abstracts,
and PDFs needed for that step, but does not label them as completed human
reference hypotheses.
