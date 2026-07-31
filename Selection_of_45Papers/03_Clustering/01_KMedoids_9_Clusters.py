"""Cluster 1,487 papers into nine semantic groups using K-medoids.

For each paper, the title and abstract are concatenated and represented with
OpenAI's text-embedding-3-large model. The archived embedding cache is used by
default, so the API is not called again unless the cache is missing.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn_extra.cluster import KMedoids


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    BUNDLE_ROOT
    / "01_Input_Data"
    / "1487_Papers_Titles_Abstracts_and_Citations.xlsx"
)
CACHE_PATH = (
    BUNDLE_ROOT
    / "02_Embedding_Cache"
    / "embeddings_text-embedding-3-large_1487x3072.npy"
)
OUTPUT_PATH = Path(__file__).with_name(
    "1487_Papers_KMedoids_9_Clusters_Reproduced.xlsx"
)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
BATCH_SIZE = 20
N_CLUSTERS = 9
RANDOM_STATE = 42


def build_embedding_texts(df: pd.DataFrame) -> list[str]:
    """Concatenate title and abstract in the original row order.

    When an abstract is unavailable, the paper title is used on its own.
    """
    titles = df["txt_title"].fillna("").astype(str).str.strip()
    abstracts = df["Abstract"].fillna("").astype(str).str.strip()
    texts = (titles + " " + abstracts).str.strip().tolist()
    if any(not text for text in texts):
        raise ValueError("At least one record has neither a title nor an abstract.")
    return texts


def request_embeddings(texts: list[str]) -> np.ndarray:
    """Request embeddings in batches and stop if any batch fails."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("The OPENAI_API_KEY environment variable is not set.")

    client = OpenAI()
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)

    embeddings = np.asarray(all_embeddings, dtype=np.float64)
    np.save(CACHE_PATH, embeddings)
    return embeddings


def validate_embeddings(embeddings: np.ndarray, expected_rows: int) -> None:
    expected_shape = (expected_rows, EMBEDDING_DIM)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Embedding shape is {embeddings.shape}; expected {expected_shape}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("The embedding matrix contains NaN or infinite values.")
    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("The embedding matrix contains one or more zero vectors.")


def main() -> None:
    df = pd.read_excel(INPUT_PATH)
    required_columns = {
        "rec_index",
        "txt_title",
        "txt_year",
        "Times Cited, WoS Core",
        "Abstract",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing input columns: {sorted(missing_columns)}")
    if len(df) != 1487:
        raise ValueError(f"Expected 1,487 input rows; found {len(df)}.")

    texts = build_embedding_texts(df)
    if CACHE_PATH.exists():
        embeddings = np.load(CACHE_PATH)
    else:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        embeddings = request_embeddings(texts)
    validate_embeddings(embeddings, len(df))

    model = KMedoids(
        n_clusters=N_CLUSTERS,
        metric="cosine",
        method="pam",
        random_state=RANDOM_STATE,
        max_iter=300,
    )
    df["cluster_id"] = model.fit_predict(embeddings)

    cluster_sizes = df["cluster_id"].value_counts().sort_index()
    if set(cluster_sizes.index) != set(range(N_CLUSTERS)):
        raise ValueError(f"Incomplete cluster IDs: {cluster_sizes.to_dict()}")

    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Saved reproduced clustering results to: {OUTPUT_PATH}")
    print(f"Cluster sizes: {cluster_sizes.to_dict()}")


if __name__ == "__main__":
    main()
