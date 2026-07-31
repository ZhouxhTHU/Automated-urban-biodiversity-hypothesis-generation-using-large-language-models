from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# User settings
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "figure"
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 300
SAVE_FORMATS = ["png", "pdf"]

SHOW_FIGURES = False


# ============================================================
# Fixed plotting settings
# ============================================================

DIMENSION_ORDER = ["Novelty", "Significance", "Testability"]

MODEL_LABELS = {
    "claude": "Claude",
    "gemini": "Gemini",
    "gpt": "GPT",
    "gpt-5.5": "GPT",
    "gemini-3.5-flash-thinking": "Gemini",
    "claude-sonnet-5": "Claude",
}

MODEL_COLORS = {
    "claude": "#015493",
    "gemini": "#019092",
    "gpt": "#F4A99B",
    "gpt-5.5": "#F4A99B",
    "gemini-3.5-flash-thinking": "#019092",
    "claude-sonnet-5": "#015493",
}

HUMAN_COLOR = "#999999"
GRID_COLOR = "#d9d9d9"


# ============================================================
# Utility functions
# ============================================================

def setup_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def find_file(filename):
    exact_path = INPUT_DIR / filename
    if exact_path.exists():
        return exact_path

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidates = sorted(
        INPUT_DIR.glob(f"{stem}*{suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Cannot find required file: {INPUT_DIR / filename}")


def read_csv(filename):
    path = find_file(filename)
    print(f"Reading: {path.name}")
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_model_name(model):
    model = str(model).strip()
    low = model.lower()

    if "claude" in low:
        return "claude"
    if "gemini" in low:
        return "gemini"
    if "gpt" in low:
        return "gpt"

    return low


def prepare_model_column(df):
    df = df.copy()
    df["model"] = df["model"].apply(clean_model_name)
    model_order = [m for m in ["claude", "gemini", "gpt"] if m in set(df["model"])]
    return df[df["model"].isin(model_order)], model_order


def model_label(model):
    return MODEL_LABELS.get(model, model)


def model_color(model):
    return MODEL_COLORS.get(model, "#7f7f7f")


def save_figure(fig, filename):
    for ext in SAVE_FORMATS:
        out_path = OUTPUT_DIR / f"{filename}.{ext}"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def panel_label(ax, label):
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left"
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def set_percent_axis(ax, ymin=0.0, ymax=1.0, step=0.2):
    ax.set_ylim(ymin, ymax)
    ticks = np.arange(ymin, ymax + 1e-9, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(round(t * 100))}%" for t in ticks])


def make_model_legend(model_order, include_human=False):
    handles = []

    if include_human:
        handles.append(
            Line2D(
                [0], [0],
                color=HUMAN_COLOR,
                marker="s",
                linewidth=1.8,
                markersize=5,
                label="Human experts"
            )
        )

    for model in model_order:
        handles.append(
            Line2D(
                [0], [0],
                color=model_color(model),
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=model_label(model)
            )
        )

    return handles


# ============================================================
# Four-panel consistency figure
# ============================================================

def plot_figure_1(overall, by_dimension, tie_behavior, by_dimension_pair_type):
    overall, model_order = prepare_model_column(overall)
    by_dimension, _ = prepare_model_column(by_dimension)
    tie_behavior, _ = prepare_model_column(tie_behavior)
    by_dimension_pair_type, _ = prepare_model_column(by_dimension_pair_type)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.6, 6.6),
        constrained_layout=True
    )

    # --------------------------------------------------------
    # b. Overall agreement
    # --------------------------------------------------------
    ax = axes[0, 1]

    metrics = [
        ("exact_agreement", "Exact"),
        ("cohen_kappa", "Kappa"),
        ("gwet_ac1", "AC1"),
        ("gwet_ac2_quadratic", "AC2"),
    ]

    x = np.arange(len(metrics))
    width = 0.22

    for i, model in enumerate(model_order):
        row = overall[overall["model"] == model].iloc[0]
        values = [float(row[col]) for col, _ in metrics]

        ax.bar(
            x + (i - (len(model_order) - 1) / 2) * width,
            values,
            width=width,
            color=model_color(model),
            edgecolor="white",
            linewidth=0.6,
            label=model_label(model)
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Agreement")
    ax.set_title("Overall agreement")
    ax.set_ylim(-0.04, 0.48)
    clean_axis(ax)
    panel_label(ax, "b")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(model_order)
    )

    # --------------------------------------------------------
    # c. Agreement by dimension
    # --------------------------------------------------------
    ax = axes[1, 0]

    x = np.arange(len(DIMENSION_ORDER))

    for model in model_order:
        data = (
            by_dimension[by_dimension["model"] == model]
            .set_index("Dimension")
            .reindex(DIMENSION_ORDER)
        )

        values = data["exact_agreement"].astype(float).values

        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=model_color(model),
            label=model_label(model)
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DIMENSION_ORDER)
    ax.set_ylabel("Exact agreement")
    ax.set_title("Agreement by dimension")
    ax.set_ylim(0.36, 0.46)
    clean_axis(ax)
    panel_label(ax, "c")
    ax.legend(frameon=False, loc="upper right")

    # --------------------------------------------------------
    # d. Tie behavior by dimension
    # --------------------------------------------------------
    ax = axes[1, 1]

    x = np.arange(len(DIMENSION_ORDER))

    human_tie = (
        tie_behavior
        .groupby("Dimension")["human_tie_rate"]
        .mean()
        .reindex(DIMENSION_ORDER)
        .astype(float)
        .values
    )

    ax.plot(
        x,
        human_tie,
        marker="s",
        linewidth=1.9,
        markersize=5,
        color=HUMAN_COLOR,
        label="Human experts"
    )

    for model in model_order:
        data = (
            tie_behavior[tie_behavior["model"] == model]
            .set_index("Dimension")
            .reindex(DIMENSION_ORDER)
        )

        values = data["llm_tie_rate"].astype(float).values

        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=model_color(model),
            label=model_label(model)
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DIMENSION_ORDER)
    ax.set_ylabel("Tie rate")
    ax.set_title("Tie behavior")
    set_percent_axis(ax, ymin=0.0, ymax=0.26, step=0.05)
    clean_axis(ax)
    panel_label(ax, "d")
    ax.legend(
        handles=make_model_legend(model_order, include_human=True),
        frameon=False,
        loc="upper left"
    )

    # --------------------------------------------------------
    # a. Source preference in HUM-vs-LLM pairs
    # --------------------------------------------------------
    ax = axes[0, 0]

    source_df = by_dimension_pair_type[
        by_dimension_pair_type["pair_type"] == "HUM-vs-LLM"
    ].copy()

    x = np.arange(len(DIMENSION_ORDER))

    human_pref = (
        source_df
        .groupby("Dimension")["human_prefers_llm_rate_in_hum_vs_llm"]
        .mean()
        .reindex(DIMENSION_ORDER)
        .astype(float)
        .values
    )

    ax.plot(
        x,
        human_pref,
        marker="s",
        linewidth=1.9,
        markersize=5,
        color=HUMAN_COLOR,
        label="Human experts"
    )

    for model in model_order:
        data = (
            source_df[source_df["model"] == model]
            .set_index("Dimension")
            .reindex(DIMENSION_ORDER)
        )

        values = data["llm_prefers_llm_rate_in_hum_vs_llm"].astype(float).values

        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=model_color(model),
            label=model_label(model)
        )

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(DIMENSION_ORDER)
    ax.set_ylabel("LLM preference rate")
    ax.set_title("Source preference")
    set_percent_axis(ax, ymin=0.0, ymax=1.0, step=0.2)
    clean_axis(ax)
    panel_label(ax, "a")
    ax.legend(
        handles=make_model_legend(model_order, include_human=True),
        frameon=False,
        loc="lower left"
    )

    save_figure(fig, "llm_human_consistency")


# ============================================================
# Main
# ============================================================

def main():
    setup_style()

    overall = read_csv("overall.csv")
    by_dimension = read_csv("by_dimension.csv")
    tie_behavior = read_csv("tie_behavior.csv")
    by_dimension_pair_type = read_csv("by_dimension_pair_type.csv")

    plot_figure_1(
        overall=overall,
        by_dimension=by_dimension,
        tie_behavior=tie_behavior,
        by_dimension_pair_type=by_dimension_pair_type
    )

    print("\nAll figures have been saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
