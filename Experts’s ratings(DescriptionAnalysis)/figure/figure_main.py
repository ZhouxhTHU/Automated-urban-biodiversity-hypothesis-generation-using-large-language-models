import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# ======================================
# Configuration
# ======================================

input_file = "../Hypothesis_Analysis_Result.xlsx"

plot_dir = "figures"
os.makedirs(plot_dir, exist_ok=True)

dimensions = ["Novelty", "Significance", "Testability"]

SOURCE_COLORS = {
    "Human": "#56B4E9",
    "LLM": "#E69F00",
    "Tie": "#CC79A7",
}

DIMENSION_COLORS = {
    "Novelty": "#56B4E9",
    "Significance": "#E69F00",
    "Testability": "#CC79A7",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ======================================
# Load the existing results from Excel
# ======================================

# Used for Figure 1: Cross_HH_vs_HL
cross_df = pd.read_excel(
    input_file,
    sheet_name="Cross_HH_vs_HL",
    index_col=0
)

# Used for Figures 2 and 3: Expert_Cross_Level
expert_cross_df = pd.read_excel(
    input_file,
    sheet_name="Expert_Cross_Level"
)


# ======================================
# Figure 1: cross-group HH vs HL selection results
# The y-axis shows the number of wins rather than proportions
# ======================================

cross_plot_df = cross_df[["HH_win", "HL_win", "Tie"]].copy()

x = np.arange(len(cross_plot_df.index))
width = 0.24

plt.figure(figsize=(7.2, 4.8))

plt.bar(
    x - width,
    cross_plot_df["HH_win"],
    width=width,
    label="Human win",
    color=SOURCE_COLORS["Human"],
    edgecolor="black",
    linewidth=0.5
)

plt.bar(
    x,
    cross_plot_df["HL_win"],
    width=width,
    label="LLM win",
    color=SOURCE_COLORS["LLM"],
    edgecolor="black",
    linewidth=0.5
)

plt.bar(
    x + width,
    cross_plot_df["Tie"],
    width=width,
    label="Tie",
    color=SOURCE_COLORS["Tie"],
    edgecolor="black",
    linewidth=0.5
)

plt.xticks(x, cross_plot_df.index)
plt.xlabel("Dimension")
plt.ylabel("Number of selections")
plt.title("Cross-source comparison outcomes")

plt.legend(frameon=False)

plt.tight_layout()

fig1_path = os.path.join(
    plot_dir,
    "figure1_cross_source_counts.png"
)

plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()


# ======================================
# Reshape into a long-form expert-by-dimension table
# Used for Figures 2 and 3
# ======================================

expert_dim_results = []

for _, row in expert_cross_df.iterrows():

    expert_id = row["Expert_ID"]

    for dim in dimensions:

        hh_win = row[f"{dim}_Cross_HH_win"]
        hl_win = row[f"{dim}_Cross_HL_win"]
        tie = row[f"{dim}_Cross_Tie"]

        total = hh_win + hl_win + tie
        decisive = hh_win + hl_win

        expert_dim_results.append({
            "Expert_ID": expert_id,
            "Dimension": dim,

            # Proportion of decisive choices in which the LLM was selected
            "LLM_Share_Among_Decisive_Dim": (
                hl_win / decisive if decisive > 0 else np.nan
            ),

            # Proportion of ties in cross-group comparisons
            "Cross_Tie_Rate_Dim": (
                tie / total if total > 0 else np.nan
            )
        })

expert_dim_df = pd.DataFrame(expert_dim_results)


# ======================================
# General-purpose function: boxplot with jittered points
# ======================================

def plot_boxplot_with_points(
    df,
    value_col,
    ylabel,
    title,
    output_path,
    ref_line=None,
    ylim=(0, 1),
    show_expert_lines=False
):
    plt.figure(figsize=(7.2, 4.8))

    positions = np.arange(1, len(dimensions) + 1)

    data = [
        df[df["Dimension"] == dim][value_col].dropna().values
        for dim in dimensions
    ]

    # Boxplot
    plt.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showfliers=False,
        boxprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"}
    )

    # Optional: connect the three dimensions for each expert
    if show_expert_lines:

        for expert_id, sub_df in df.sourceby("Expert_ID"):

            sub_df = sub_df.set_index("Dimension").reindex(dimensions)

            y_values = sub_df[value_col].values

            if np.all(pd.notna(y_values)):
                plt.plot(
                    positions,
                    y_values,
                    linewidth=0.8,
                    alpha=0.25
                )

    # Jittered points: each point represents one expert
    rng = np.random.default_rng(2026)

    for pos, dim in zip(positions, dimensions):

        values = df[df["Dimension"] == dim][value_col].dropna().values

        jitter = rng.normal(
            loc=0,
            scale=0.04,
            size=len(values)
        )

        plt.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=36,
            color=DIMENSION_COLORS[dim],
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85
        )

    # Reference line
    if ref_line is not None:
        plt.axhline(
            y=ref_line,
            linewidth=1,
            linestyle="--"
        )

    plt.xticks(positions, dimensions)
    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ======================================
# Figure 2: boxplot of expert-level LLM share across the three dimensions
# ======================================

fig2_path = os.path.join(
    plot_dir,
    "figure2_expert_llm_share_by_dimension_boxplot.png"
)

# Figure 2: LLM share
plot_boxplot_with_points(
    df=expert_dim_df,
    value_col="LLM_Share_Among_Decisive_Dim",
    ylabel="LLM share among decisive choices",
    title="Expert-level LLM preference by dimension",
    output_path=fig2_path,
    ref_line=0.5,
    ylim=(0, 1),
    show_expert_lines=False
)

# ======================================
# Figure 3: boxplot of expert-level tie rate across the three dimensions
# ======================================

fig3_path = os.path.join(
    plot_dir,
    "figure3_expert_tie_rate_by_dimension_boxplot.png"
)

# Figure 3: Tie rate
plot_boxplot_with_points(
    df=expert_dim_df,
    value_col="Cross_Tie_Rate_Dim",
    ylabel="Cross-source Tie rate",
    title="Expert-level Tie rate by dimension",
    output_path=fig3_path,
    ref_line=None,
    ylim=(0, 1),
    show_expert_lines=False
)


# ======================================
# Save the processed expert-by-dimension data
# ======================================

expert_dim_data_path = os.path.join(
    plot_dir,
    "expert_dimension_plot_data.xlsx"
)

expert_dim_df.to_excel(
    expert_dim_data_path,
    index=False
)


# ======================================
# Print the output paths
# ======================================

print("\nSaved figures:")
print(fig1_path)
print(fig2_path)
print(fig3_path)

print("\nSaved plot data:")
print(expert_dim_data_path)


# ======================================
# Create the combined a/b/c figure
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

PANEL_FILES = [
    ("a", "figure1_cross_source_counts.png"),
    ("b", "figure2_expert_llm_share_by_dimension_boxplot.png"),
    ("c", "figure3_expert_tie_rate_by_dimension_boxplot.png"),
]

CAPTION = (
    "Figure X. Descriptive outcomes of cross-source comparisons. "
    "a, Numbers of judgments favoring human-proposed hypotheses, "
    "LLM-generated hypotheses, or a tie in each evaluation dimension. "
    "b, Expert-level shares of LLM selections among decisive cross-source "
    "judgments. c, Expert-level tie rates in cross-source judgments."
)


def add_panel(ax, image_path, label):
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        label,
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=3,
    )


def combine_figures():
    panel_paths = [
        (label, os.path.join(FIGURE_DIR, filename))
        for label, filename in PANEL_FILES
    ]

    missing_files = [
        image_path
        for _, image_path in panel_paths
        if not os.path.exists(image_path)
    ]

    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing input figure(s):\n{missing}")

    fig = plt.figure(figsize=(10.5, 8.8))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        hspace=0.05,
        wspace=0.03,
    )

    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    axes = [ax_a, ax_b, ax_c]

    for ax, (label, image_path) in zip(axes, panel_paths):
        add_panel(ax, image_path, label)

    output_png = os.path.join(
        FIGURE_DIR,
        "figureX_descriptive_outcomes_cross_source_comparisons.png",
    )
    output_pdf = os.path.join(
        FIGURE_DIR,
        "figureX_descriptive_outcomes_cross_source_comparisons.pdf",
    )
    caption_file = os.path.join(
        FIGURE_DIR,
        "figureX_descriptive_outcomes_cross_source_comparisons_caption.txt",
    )

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    with open(caption_file, "w", encoding="utf-8") as file:
        file.write(textwrap.fill(CAPTION, width=100))
        file.write("\n")

    print("\nSaved combined figure:")
    print(output_png)
    print(output_pdf)
    print("\nSaved caption:")
    print(caption_file)


combine_figures()
