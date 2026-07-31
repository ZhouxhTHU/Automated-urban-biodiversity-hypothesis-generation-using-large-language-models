# Script for plotting open-ended expert feedback without rerunning API.
# It reads the already coded file: coded_responses_empirical.csv
# and combines sentiment, pros, and cons into one a/b/c figure.

import os
import ast
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

# ----------------------------------------------------------
# Compatibility patch:
# This keeps seaborn usable with newer matplotlib versions.
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
            pass

    cm.register_cmap = _register_cmap_compat

import seaborn as sns
from thefuzz import process, fuzz


# ==========================================
# 1. Configuration
# ==========================================

OUTPUT_DIR = "./Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 优先读取 Outputs 里的 coded_responses_empirical.csv
# 如果没有，则读取当前文件夹下的 coded_responses_empirical.csv
CANDIDATE_CODED_FILES = [
    os.path.join(OUTPUT_DIR, "coded_responses_empirical.csv"),
    "coded_responses_empirical.csv"
]

OUTPUT_STEM = "open_feedback_abc"

SENTIMENT_ORDER = ["Positive", "Mixed", "Neutral", "Negative"]


# ==========================================
# 2. Load coded data
# ==========================================

def find_coded_file() -> str:
    for file_path in CANDIDATE_CODED_FILES:
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(
        "Cannot find coded_responses_empirical.csv. "
        "Please put it either in ./Outputs/ or in the current working directory."
    )


def parse_list_cell(value):
    """
    Convert a CSV cell back to a Python list.

    The coded CSV usually stores pros and cons as strings like:
    "['Highly Novel', 'Structured']"
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        if value == "" or value.lower() == "nan":
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)]
        except Exception:
            # Fallback: treat as one item
            return [value]

    return [str(value)]


def load_coded_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required_cols = [
        "sentiment_category",
        "pros",
        "cons"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns in {file_path}: {missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    df["pros"] = df["pros"].apply(parse_list_cell)
    df["cons"] = df["cons"].apply(parse_list_cell)

    return df


# ==========================================
# 3. Semantic clustering
# ==========================================

def group_synonyms(items: list[str], threshold: int = 75) -> list[tuple[str, int]]:
    if not items:
        return []

    cleaned_items = [
        item.strip().title()
        for item in items
        if item is not None and str(item).strip() != ""
    ]

    if not cleaned_items:
        return []

    raw_counts = Counter(cleaned_items)
    sorted_unique_items = [
        item for item, count in raw_counts.most_common()
    ]

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

    clustered_items = [
        canonical_mapping[item]
        for item in cleaned_items
    ]

    return Counter(clustered_items).most_common(10)


# ==========================================
# 4. Plotting
# ==========================================

def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left"
    )


def plot_bar_values_vertical(ax):
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.08,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9
        )


def plot_bar_values_horizontal(ax):
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + 0.04,
            y,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=8.5
        )


def visualize_findings_abc(df: pd.DataFrame, output_dir: str = "./Outputs"):
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans"
            ],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    # Sentiment counts
    sentiment_counts = (
        df["sentiment_category"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .reset_index()
    )
    sentiment_counts.columns = ["sentiment_category", "count"]

    # Pros and cons counts
    all_pros = [
        item
        for sublist in df["pros"]
        for item in sublist
    ]

    all_cons = [
        item
        for sublist in df["cons"]
        for item in sublist
    ]

    pros_counts = group_synonyms(all_pros, threshold=75)
    cons_counts = group_synonyms(all_cons, threshold=75)

    pros_df = pd.DataFrame(
        pros_counts,
        columns=["descriptor", "frequency"]
    )

    cons_df = pd.DataFrame(
        cons_counts,
        columns=["descriptor", "frequency"]
    )

    # Save counts for checking
    counts_output_path = os.path.join(
        output_dir,
        f"{OUTPUT_STEM}_counts.csv"
    )

    counts_for_save = pd.concat(
        [
            sentiment_counts.assign(panel="a_sentiment").rename(
                columns={
                    "sentiment_category": "descriptor",
                    "count": "frequency"
                }
            )[["panel", "descriptor", "frequency"]],
            pros_df.assign(panel="b_pros")[["panel", "descriptor", "frequency"]],
            cons_df.assign(panel="c_cons")[["panel", "descriptor", "frequency"]],
        ],
        ignore_index=True
    )

    counts_for_save.to_csv(
        counts_output_path,
        index=False
    )

    # ==========================================
    # Combined a/b/c figure
    # Layout:
    # a: top, spanning two columns
    # b: bottom-left
    # c: bottom-right
    # ==========================================

    fig = plt.figure(figsize=(12.5, 8.8))

    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[0.85, 1.35],
        width_ratios=[1, 1],
        hspace=0.45,
        wspace=0.55
    )

    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    # Panel a: sentiment distribution
    sentiment_palette = {
        "Positive": "#009E73",
        "Mixed": "#E69F00",
        "Neutral": "#7F7F7F",
        "Negative": "#D55E00"
    }

    sns.barplot(
        data=sentiment_counts,
        x="sentiment_category",
        y="count",
        order=SENTIMENT_ORDER,
        palette=sentiment_palette,
        saturation=1,
        ax=ax_a
    )

    ax_a.set_title(
        "Distribution of Expert Sentiment Toward LLM Hypotheses",
        pad=10
    )
    ax_a.set_xlabel("Sentiment Category")
    ax_a.set_ylabel("Number of Experts")
    ax_a.set_ylim(
        0,
        max(sentiment_counts["count"].max() + 1, 1)
    )

    #plot_bar_values_vertical(ax_a)
    add_panel_label(ax_a, "a", x=-0.055, y=1.13)

    # Panel b: pros
    if not pros_df.empty:
        sns.barplot(
            data=pros_df,
            x="frequency",
            y="descriptor",
            ax=ax_b,
            color="#0072B2",
            saturation=1
        )

    ax_b.set_title("Most Frequently Cited Pros")
    ax_b.set_xlabel("Frequency")
    ax_b.set_ylabel("")
    ax_b.set_xlim(
        0,
        max(
            2.2,
            pros_df["frequency"].max() + 0.45
            if not pros_df.empty else 2.2
        )
    )

    #plot_bar_values_horizontal(ax_b)
    add_panel_label(ax_b, "b", x=-0.24, y=1.08)

    # Panel c: cons
    if not cons_df.empty:
        sns.barplot(
            data=cons_df,
            x="frequency",
            y="descriptor",
            ax=ax_c,
            color="#D55E00",
            saturation=1
        )

    ax_c.set_title("Most Frequently Cited Cons")
    ax_c.set_xlabel("Frequency")
    ax_c.set_ylabel("")
    ax_c.set_xlim(
        0,
        max(
            2.2,
            cons_df["frequency"].max() + 0.45
            if not cons_df.empty else 2.2
        )
    )

    #plot_bar_values_horizontal(ax_c)
    add_panel_label(ax_c, "c", x=-0.24, y=1.08)

    # Make grid less visually heavy
    for ax in [ax_a, ax_b, ax_c]:
        ax.grid(axis="y", color="0.90", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # For horizontal bar charts, vertical grid is more useful
    for ax in [ax_b, ax_c]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.grid(axis="x", color="0.90", linewidth=0.7)
        ax.grid(axis="y", visible=False)

    png_path = os.path.join(
        output_dir,
        f"{OUTPUT_STEM}.png"
    )

    pdf_path = os.path.join(
        output_dir,
        f"{OUTPUT_STEM}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved combined figure:")
    print(png_path)
    print(pdf_path)

    print("\nSaved counts table:")
    print(counts_output_path)


# ==========================================
# 5. Main
# ==========================================

if __name__ == "__main__":

    coded_file = find_coded_file()

    print(f"Loading coded data from: {coded_file}")

    analyzed_df = load_coded_data(coded_file)

    print(f"Loaded {len(analyzed_df)} coded responses.")

    visualize_findings_abc(
        df=analyzed_df,
        output_dir=OUTPUT_DIR
    )
