# Script for analyzing the answers of experts to open-ended questions regarding LLM-generated hypotheses.
# Authors: Jun Yang, Google Gemini; Date: 29 June 2026
# Modified: only API-calling logic is changed to OpenAI-compatible style.
# Visualization logic is kept consistent with the original seaborn-based version.

import os
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Compatibility patch:
# This keeps the original seaborn plotting code usable when
# seaborn is used with newer matplotlib versions.
# It does not change the plotting style or colors.
# ----------------------------------------------------------
import matplotlib as mpl
import matplotlib.cm as cm

if not hasattr(cm, "register_cmap"):
    def _register_cmap_compat(name=None, cmap=None, **kwargs):
        if cmap is None:
            cmap = kwargs.get("cmap", None)

        if cmap is None:
            return

        if name is None:
            name = getattr(cmap, "name", None)

        try:
            mpl.colormaps.register(cmap, name=name, force=True)
        except ValueError:
            # The colormap may already be registered.
            pass

    cm.register_cmap = _register_cmap_compat

import seaborn as sns

from collections import Counter
from pydantic import BaseModel, Field
from typing import Literal
from thefuzz import process, fuzz
from openai import OpenAI


# ==========================================
# 0. API Configuration
# ==========================================
Expert_MODEL_CONFIG = {
    "config_name": "Expert_config",
    "model_type": "openai_chat",
    "model_name": "gemini-3.5-flash-thinking",
    "api_key": os.getenv("EXPERT_API_KEY", ""),
    "client_args": {
        # "base_url": "https://svip.xty.app/v1",
        "base_url": "https://svip-ip.xty.app/v1",
        "timeout": 600
    },
    "generation_kwargs": {
        "response_format": {
            "type": "json_object"
        },
        "temperature": 0.0
    }
}


# ==========================================
# 1. Structured Data Schema (Pydantic)
# ==========================================
class QualitativeCoding(BaseModel):
    sentiment_category: Literal["Positive", "Neutral", "Negative", "Mixed"]
    sentiment_score: int = Field(
        description="Sentiment score from 1 (Very Negative) to 5 (Very Positive)"
    )
    primary_topics: list[str] = Field(
        description="1 to 3 core themes discussed (e.g., 'Literature Synthesis', 'Field Reality')"
    )
    pros: list[str] = Field(
        description="Specific advantages or strengths cited regarding LLM hypotheses (1-3 words max per item)"
    )
    cons: list[str] = Field(
        description="Specific limitations, risks, or weaknesses cited regarding LLM hypotheses (1-3 words max per item)"
    )


# ==========================================
# 2. Logger
# ==========================================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("OpinionAnalysis")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# ==========================================
# 3. Data Loader
# ==========================================
def load_empirical_data(file_path: str, id_col: str, feedback_col: str) -> pd.DataFrame:
    """
    Loads real survey data from an Excel or CSV file and standardizes it for the pipeline.
    """
    print(f"Loading data from: {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[!] Could not find the file at {file_path}")

    # Read Excel or CSV based on file extension
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    # Check if the specified columns actually exist in the file
    if id_col not in df.columns or feedback_col not in df.columns:
        raise KeyError(
            f"[!] Missing columns. Ensure your file has '{id_col}' and '{feedback_col}'. "
            f"Found: {list(df.columns)}"
        )

    # Rename columns to standardized internal names
    df = df.rename(columns={id_col: 'expert_id', feedback_col: 'feedback'})

    # Clean the data: drop rows where the participant didn't leave any feedback
    original_len = len(df)
    df = df.dropna(subset=['feedback'])
    df['feedback'] = df['feedback'].astype(str)
    df = df[df['feedback'].str.strip() != '']

    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} empty responses.")

    # Ensure expert_id is a string (prevents pandas merging errors later)
    df['expert_id'] = df['expert_id'].astype(str)

    return df


# ==========================================
# 4. OpenAI-Compatible API Engine
# ==========================================
def create_openai_compatible_client(model_config: dict) -> OpenAI:
    """
    Creates an OpenAI-compatible client using the model_config dictionary.
    """
    client_args = model_config.get("client_args", {})

    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=client_args.get("base_url"),
        timeout=client_args.get("timeout", 600)
    )

    return client


def clean_json_text(text: str) -> str:
    """
    Removes possible Markdown code fences around JSON output.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def make_api_call_with_retry(
    prompt: str,
    logger: logging.Logger,
    model_config: dict,
    max_retries: int = 3,
    sleep_seconds: int = 5
) -> str:
    """
    Encapsulates OpenAI-compatible API calls.
    This replaces the original google.genai direct call, while keeping the rest of the pipeline unchanged.
    """
    last_error = None

    while True:
        for attempt in range(1, max_retries + 1):
            try:
                client = create_openai_compatible_client(model_config)

                generation_kwargs = model_config.get("generation_kwargs", {}).copy()

                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert qualitative researcher analyzing survey data from "
                                "scientists evaluating hypotheses generated by Large Language Models. "
                                "Return valid JSON only. Do not include Markdown, explanation, or extra text."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    **generation_kwargs
                )

                content = response.choices[0].message.content

                if not content:
                    raise ValueError("Empty response from API.")

                return clean_json_text(content)

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                logger.error(
                    f"The API call failed on attempt {attempt}/{max_retries}. Error: {e}"
                )

                if model_config.get("api_key"):
                    logger.info(f"Current API Key: ...{model_config['api_key'][-4:]}")

                if attempt < max_retries:
                    logger.info(f"Will retry the failed API call after {sleep_seconds} seconds...")
                    time.sleep(sleep_seconds)

        raise RuntimeError(
            f"API call failed after {max_retries} attempts. Last error: {last_error}"
        )


# ==========================================
# 5. LLM Extraction Engine
# ==========================================
def extract_qualitative_data_gemini(df: pd.DataFrame, model_config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Function name is kept close to the original code.
    The internal API call has been changed from google.genai to OpenAI-compatible API.
    """
    print("Initializing Gemini extraction pipeline...")

    extracted_data = []

    for index, row in df.iterrows():
        print(f"Processing Expert {row['expert_id']}...")

        prompt = (
            "You are an expert qualitative researcher analyzing survey data from "
            "scientists evaluating hypotheses generated by Large Language Models. "
            "Read the following feedback and extract the sentiment, topics, pros, and cons.\n\n"
            "Return only one valid JSON object with exactly the following fields:\n"
            "{\n"
            '  "sentiment_category": "Positive, Neutral, Negative, or Mixed",\n'
            '  "sentiment_score": 1,\n'
            '  "primary_topics": ["topic 1", "topic 2"],\n'
            '  "pros": ["short pro 1", "short pro 2"],\n'
            '  "cons": ["short con 1", "short con 2"]\n'
            "}\n\n"
            "Rules:\n"
            "1. sentiment_category must be one of: Positive, Neutral, Negative, Mixed.\n"
            "2. sentiment_score must be an integer from 1 to 5.\n"
            "3. primary_topics should contain 1 to 3 core themes.\n"
            "4. pros should contain specific advantages or strengths. Each item should be 1 to 3 words.\n"
            "5. cons should contain specific limitations, risks, or weaknesses. Each item should be 1 to 3 words.\n"
            "6. If no clear pro or con is mentioned, return an empty list for that field.\n"
            "7. Do not include Markdown code fences.\n"
            "8. Do not include explanation.\n\n"
            f"Feedback: \"{row['feedback']}\""
        )

        try:
            response_text = make_api_call_with_retry(
                prompt=prompt,
                logger=logger,
                model_config=model_config,
                max_retries=3,
                sleep_seconds=5
            )

            coding = QualitativeCoding.model_validate_json(response_text)

            extracted_data.append({
                "expert_id": row['expert_id'],
                "sentiment_category": coding.sentiment_category,
                "sentiment_score": coding.sentiment_score,
                "primary_topics": coding.primary_topics,
                "pros": coding.pros,
                "cons": coding.cons
            })

        except Exception as e:
            print(f"Error processing {row['expert_id']}: {e}")
            logger.error(f"Error processing Expert {row['expert_id']}: {e}", exc_info=True)
            extracted_data.append(None)

    extracted_df = pd.DataFrame([d for d in extracted_data if d is not None])

    if extracted_df.empty:
        raise RuntimeError("No valid LLM-coded responses were produced.")

    final_df = pd.merge(df, extracted_df, on="expert_id")
    return final_df


# ==========================================
# 6. Semantic Clustering (Fuzzy Matching)
# ==========================================
def group_synonyms(items: list[str], threshold: int = 75) -> list[tuple[str, int]]:
    if not items:
        return []

    cleaned_items = [item.strip().title() for item in items if item is not None and str(item).strip() != ""]

    if not cleaned_items:
        return []

    raw_counts = Counter(cleaned_items)
    sorted_unique_items = [item for item, count in raw_counts.most_common()]

    canonical_mapping = {}

    for item in sorted_unique_items:
        established_terms = list(set(canonical_mapping.values()))
        if established_terms:
            best_match, score = process.extractOne(
                item,
                established_terms,
                scorer=fuzz.token_sort_ratio
            )

            if score >= threshold:
                canonical_mapping[item] = best_match
                continue

        canonical_mapping[item] = item

    clustered_items = [canonical_mapping[item] for item in cleaned_items]
    return Counter(clustered_items).most_common(10)


# ==========================================
# 7. Aggregation & Visualization
# ==========================================
def visualize_findings(df: pd.DataFrame, output_dir: str = "./Outputs"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Sentiment Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="sentiment_category",
        order=["Positive", "Mixed", "Neutral", "Negative"],
        palette="muted"
    )
    plt.title("Distribution of Expert Sentiment Toward LLM Hypotheses")
    plt.ylabel("Number of Experts")
    plt.xlabel("Sentiment Category")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    all_pros = [item for sublist in df['pros'] for item in sublist]
    all_cons = [item for sublist in df['cons'] for item in sublist]

    pros_counts = group_synonyms(all_pros, threshold=75)
    cons_counts = group_synonyms(all_cons, threshold=75)

    # 2. Plot Clustered Pros and Cons
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if pros_counts:
        pros_labels, pros_vals = zip(*pros_counts)
        sns.barplot(x=list(pros_vals), y=list(pros_labels), ax=axes[0], color="#2ca02c")
        axes[0].set_title("Most Frequently Cited Pros (Clustered)")
        axes[0].set_xlabel("Frequency")

    if cons_counts:
        cons_labels, cons_vals = zip(*cons_counts)
        sns.barplot(x=list(cons_vals), y=list(cons_labels), ax=axes[1], color="#d62728")
        axes[1].set_title("Most Frequently Cited Cons (Clustered)")
        axes[1].set_xlabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pros_cons_frequencies.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}/")


# ==========================================
# Main Execution Pipeline
# ==========================================
if __name__ == "__main__":

    logger = setup_logger()

    if Expert_MODEL_CONFIG["api_key"] == "PASTE_YOUR_API_KEY_HERE":
        raise ValueError(
            "Please replace 'PASTE_YOUR_API_KEY_HERE' with your actual API key in Expert_MODEL_CONFIG."
        )

    # ---------------------------------------------------------
    # CONFIGURATION: Update these variables for your data!
    # ---------------------------------------------------------
    DATA_FILE = "Expertopinions.csv"         # Path to your Excel or CSV file
    ID_COLUMN_NAME = "Expert_ID"            # Exact name of the ID column in your file
    FEEDBACK_COLUMN_NAME = "Feedback"       # Exact name of the open-ended text column
    # ---------------------------------------------------------

    try:
        # 1. Load Real Data
        df = load_empirical_data(
            file_path=DATA_FILE,
            id_col=ID_COLUMN_NAME,
            feedback_col=FEEDBACK_COLUMN_NAME
        )
        print(f"Successfully loaded {len(df)} open-ended responses for analysis.")

        # 2. Extract Data using OpenAI-compatible API
        analyzed_df = extract_qualitative_data_gemini(
            df=df,
            model_config=Expert_MODEL_CONFIG,
            logger=logger
        )

        # Save raw coded data for manual review and manuscript appendix
        os.makedirs("./Outputs", exist_ok=True)
        output_csv = "./Outputs/coded_responses_empirical.csv"
        analyzed_df.to_csv(output_csv, index=False)
        print(f"\nSaved raw coded dataset to {output_csv}")

        # 3. Apply Fuzzy Matching & Visualize
        visualize_findings(analyzed_df)

    except Exception as e:
        print(f"\n[!] Pipeline halted due to an error: {e}")
        logger.error(f"Pipeline halted due to an error: {e}", exc_info=True)