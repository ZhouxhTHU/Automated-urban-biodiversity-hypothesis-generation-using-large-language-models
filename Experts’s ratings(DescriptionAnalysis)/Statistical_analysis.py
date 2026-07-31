import pandas as pd
import glob
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


# ======================================
# Configuration
# ======================================
assignment_file = "All_Assignment_Pairs.xlsx"
survey_pattern = os.path.join(
    "Expert_Results",
    "*Urban Biodiversity Hypothesis Project.xlsx"
)


# ======================================
# Load the assignment table
# ======================================
assignment_df = pd.read_excel(assignment_file)

assignment_df = assignment_df.sort_values(
    ["Expert_ID", "Pair_Number"]
).reset_index(drop=True)


# ======================================
# Initialize the statistics counters
# ======================================
dimensions = ["Novelty", "Significance", "Testability"]


def init_stat():
    return {
        dim: {
            "HH_win": 0,
            "HL_win": 0,
            "Tie": 0
        }
        for dim in dimensions
    }


# Overall
overall_stats = init_stat()

# Cross-group HH vs HL
cross_group_stats = init_stat()

# HH vs HH
hh_internal_stats = init_stat()

# HL vs HL
hl_internal_stats = init_stat()


# Hypothesis-level statistics
hypothesis_stats = defaultdict(
    lambda: {
        "comparisons": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0
    }
)


# ======================================
# Hypothesis-level statistics by dimension
# ======================================
hypothesis_dim_stats = defaultdict(
    lambda: {
        dim: {
            "comparisons": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0
        }
        for dim in dimensions
    }
)


# ======================================
# Expert-level statistics: all comparisons
# ======================================
expert_stats = defaultdict(init_stat)


# ======================================
# Added: expert-level statistics for cross-group HH vs HL comparisons only
# Used for Figure 3
# ======================================
expert_cross_stats = defaultdict(init_stat)


# ======================================
# Basic counts
# ======================================
total_experts = 0
total_pairs = 0
total_comparisons = 0


# ======================================
# Load the survey files
# ======================================
survey_files = glob.glob(survey_pattern)

print(f"Found {len(survey_files)} files")


# ======================================
# Begin processing
# ======================================
for survey_file in survey_files:

    expert_id = os.path.basename(survey_file).split()[0]

    expert_pairs = assignment_df[
        assignment_df["Expert_ID"] == expert_id
    ].sort_values("Pair_Number")

    if len(expert_pairs) == 0:
        continue

    survey_df = pd.read_excel(survey_file)

    # The second row contains the actual responses
    response_row = survey_df.iloc[1]

    question_cols = [
        col for col in survey_df.columns
        if str(col).startswith("Which hypothesis")
    ]

    expected_num = len(expert_pairs) * 3

    if len(question_cols) < expected_num:
        print(f"Skip {expert_id}: incomplete")
        continue

    total_experts += 1
    total_pairs += len(expert_pairs)
    total_comparisons += len(expert_pairs) * 3

    # Iterate over pairs
    for i, (_, pair) in enumerate(expert_pairs.iterrows()):

        hypA = pair["Hypothesis_A_ID"]
        hypB = pair["Hypothesis_B_ID"]

        pair_cols = question_cols[i*3:(i+1)*3]

        # Determine the pair type
        if hypA.startswith("HH") and hypB.startswith("HL"):
            pair_type = "cross"

        elif hypA.startswith("HL") and hypB.startswith("HH"):
            pair_type = "cross"

        elif hypA.startswith("HH") and hypB.startswith("HH"):
            pair_type = "hh_internal"

        elif hypA.startswith("HL") and hypB.startswith("HL"):
            pair_type = "hl_internal"

        else:
            pair_type = "unknown"

        for j, col in enumerate(pair_cols):

            dimension = dimensions[j]
            answer = response_row[col]

            # Hypothesis-level overall statistics
            hypothesis_stats[hypA]["comparisons"] += 1
            hypothesis_stats[hypB]["comparisons"] += 1

            # Hypothesis-level statistics by dimension
            hypothesis_dim_stats[hypA][dimension]["comparisons"] += 1
            hypothesis_dim_stats[hypB][dimension]["comparisons"] += 1

            if pd.isna(answer):
                continue

            # Tie
            if "Tie" in str(answer):

                overall_stats[dimension]["Tie"] += 1

                if pair_type == "cross":
                    cross_group_stats[dimension]["Tie"] += 1
                    expert_cross_stats[expert_id][dimension]["Tie"] += 1

                elif pair_type == "hh_internal":
                    hh_internal_stats[dimension]["Tie"] += 1

                elif pair_type == "hl_internal":
                    hl_internal_stats[dimension]["Tie"] += 1

                hypothesis_stats[hypA]["ties"] += 1
                hypothesis_stats[hypB]["ties"] += 1

                hypothesis_dim_stats[hypA][dimension]["ties"] += 1
                hypothesis_dim_stats[hypB][dimension]["ties"] += 1

                expert_stats[expert_id][dimension]["Tie"] += 1

                continue

            # Determine the winner
            if answer == "Hypothesis A":
                winner = hypA
                loser = hypB

            elif answer == "Hypothesis B":
                winner = hypB
                loser = hypA

            else:
                continue

            # Overall statistics
            if winner.startswith("HH"):
                overall_stats[dimension]["HH_win"] += 1
            else:
                overall_stats[dimension]["HL_win"] += 1

            # Statistics by comparison category
            if pair_type == "cross":

                if winner.startswith("HH"):
                    cross_group_stats[dimension]["HH_win"] += 1
                    expert_cross_stats[expert_id][dimension]["HH_win"] += 1
                else:
                    cross_group_stats[dimension]["HL_win"] += 1
                    expert_cross_stats[expert_id][dimension]["HL_win"] += 1

            elif pair_type == "hh_internal":

                hh_internal_stats[dimension]["HH_win"] += 1

            elif pair_type == "hl_internal":

                hl_internal_stats[dimension]["HL_win"] += 1

            # Hypothesis-level overall statistics
            hypothesis_stats[winner]["wins"] += 1
            hypothesis_stats[loser]["losses"] += 1

            # Hypothesis-level statistics by dimension
            hypothesis_dim_stats[winner][dimension]["wins"] += 1
            hypothesis_dim_stats[loser][dimension]["losses"] += 1

            # Expert-level HH/HL selection statistics across all comparisons
            if winner.startswith("HH"):
                expert_stats[expert_id][dimension]["HH_win"] += 1
            else:
                expert_stats[expert_id][dimension]["HL_win"] += 1


# ======================================
# Hypothesis-level results
# ======================================
hypothesis_results = []

for hyp_id, stat in hypothesis_stats.items():

    comparisons = stat["comparisons"]
    wins = stat["wins"]
    losses = stat["losses"]
    ties = stat["ties"]

    strict_win_rate = wins / comparisons if comparisons > 0 else 0

    # A statistic:
    # A = P(X1 > X2) + 0.5 * P(X1 = X2)
    a_score_rate = (wins + 0.5 * ties) / comparisons if comparisons > 0 else 0

    decisive_comparisons = wins + losses
    non_tie_win_rate = wins / decisive_comparisons if decisive_comparisons > 0 else 0

    hypothesis_results.append({
        "Hypothesis_ID": hyp_id,
        "Group": hyp_id[:2],
        "Comparisons": comparisons,
        "Wins": wins,
        "Losses": losses,
        "Ties": ties,
        "Strict_Win_Rate": round(strict_win_rate, 4),
        "A_Score_Rate": round(a_score_rate, 4),
        "NonTie_Win_Rate": round(non_tie_win_rate, 4)
    })

hypothesis_df = pd.DataFrame(hypothesis_results)

hypothesis_df = hypothesis_df.sort_values(
    ["A_Score_Rate", "Wins"],
    ascending=False
).reset_index(drop=True)


# ======================================
# Hypothesis-level results by dimension
# ======================================
hypothesis_dim_results = []

for hyp_id, dim_stat in hypothesis_dim_stats.items():

    for dim in dimensions:

        comparisons = dim_stat[dim]["comparisons"]
        wins = dim_stat[dim]["wins"]
        losses = dim_stat[dim]["losses"]
        ties = dim_stat[dim]["ties"]

        strict_win_rate = wins / comparisons if comparisons > 0 else 0

        # A statistic:
        # A = P(X1 > X2) + 0.5 * P(X1 = X2)
        a_score_rate = (wins + 0.5 * ties) / comparisons if comparisons > 0 else 0

        decisive_comparisons = wins + losses
        non_tie_win_rate = wins / decisive_comparisons if decisive_comparisons > 0 else 0

        hypothesis_dim_results.append({
            "Dimension": dim,
            "Hypothesis_ID": hyp_id,
            "Group": hyp_id[:2],
            "Comparisons": comparisons,
            "Wins": wins,
            "Losses": losses,
            "Ties": ties,
            "Strict_Win_Rate": round(strict_win_rate, 4),
            "A_Score_Rate": round(a_score_rate, 4),
            "NonTie_Win_Rate": round(non_tie_win_rate, 4)
        })

hypothesis_dim_df = pd.DataFrame(hypothesis_dim_results)

hypothesis_dim_df = hypothesis_dim_df.sort_values(
    ["Dimension", "A_Score_Rate", "Wins"],
    ascending=[True, False, False]
).reset_index(drop=True)


# ======================================
# Organize expert-level results: all comparisons
# ======================================
expert_results = []

for expert_id, stat in expert_stats.items():

    row = {"Expert_ID": expert_id}

    expert_hh_total = 0
    expert_hl_total = 0
    expert_tie_total = 0

    for dim in dimensions:

        hh_win = stat[dim]["HH_win"]
        hl_win = stat[dim]["HL_win"]
        tie = stat[dim]["Tie"]

        row[f"{dim}_HH_win"] = hh_win
        row[f"{dim}_HL_win"] = hl_win
        row[f"{dim}_Tie"] = tie

        expert_hh_total += hh_win
        expert_hl_total += hl_win
        expert_tie_total += tie

    row["Total_HH_win"] = expert_hh_total
    row["Total_HL_win"] = expert_hl_total
    row["Total_Tie"] = expert_tie_total
    row["Total_Comparisons"] = expert_hh_total + expert_hl_total + expert_tie_total

    expert_results.append(row)

expert_df = pd.DataFrame(expert_results)

if not expert_df.empty:
    expert_df = expert_df.sort_values("Expert_ID").reset_index(drop=True)


# ======================================
# Added: organize expert-level results for cross-group HH vs HL comparisons only
# Used for a more accurate analysis of direct Human vs LLM selections
# ======================================
expert_cross_results = []

for expert_id, stat in expert_cross_stats.items():

    row = {"Expert_ID": expert_id}

    cross_hh_total = 0
    cross_hl_total = 0
    cross_tie_total = 0

    for dim in dimensions:

        hh_win = stat[dim]["HH_win"]
        hl_win = stat[dim]["HL_win"]
        tie = stat[dim]["Tie"]

        row[f"{dim}_Cross_HH_win"] = hh_win
        row[f"{dim}_Cross_HL_win"] = hl_win
        row[f"{dim}_Cross_Tie"] = tie

        cross_hh_total += hh_win
        cross_hl_total += hl_win
        cross_tie_total += tie

    cross_total = cross_hh_total + cross_hl_total + cross_tie_total
    decisive_total = cross_hh_total + cross_hl_total

    row["Cross_HH_win"] = cross_hh_total
    row["Cross_HL_win"] = cross_hl_total
    row["Cross_Tie"] = cross_tie_total
    row["Cross_Total_Comparisons"] = cross_total
    row["Cross_Decisive_Comparisons"] = decisive_total

    row["Cross_Tie_Rate"] = (
        cross_tie_total / cross_total
        if cross_total > 0 else np.nan
    )

    row["LLM_Share_Among_Decisive"] = (
        cross_hl_total / decisive_total
        if decisive_total > 0 else np.nan
    )

    row["Human_Share_Among_Decisive"] = (
        cross_hh_total / decisive_total
        if decisive_total > 0 else np.nan
    )

    expert_cross_results.append(row)

expert_cross_df = pd.DataFrame(expert_cross_results)

if not expert_cross_df.empty:
    expert_cross_df = expert_cross_df.sort_values("Expert_ID").reset_index(drop=True)


# ======================================
# Convert to DataFrames
# ======================================
overall_df = pd.DataFrame(overall_stats).T
cross_df = pd.DataFrame(cross_group_stats).T
hh_df = pd.DataFrame(hh_internal_stats).T
hl_df = pd.DataFrame(hl_internal_stats).T


# ======================================
# Summary information
# ======================================
summary_df = pd.DataFrame({
    "Metric": [
        "Total Experts",
        "Total Pairs",
        "Total Comparisons"
    ],
    "Value": [
        total_experts,
        total_pairs,
        total_comparisons
    ]
})


# ======================================
# Save the Excel workbook
# ======================================
with pd.ExcelWriter("Hypothesis_Analysis_Result.xlsx") as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    overall_df.to_excel(
        writer,
        sheet_name="Overall"
    )

    cross_df.to_excel(
        writer,
        sheet_name="Cross_HH_vs_HL"
    )

    hh_df.to_excel(
        writer,
        sheet_name="HH_vs_HH"
    )

    hl_df.to_excel(
        writer,
        sheet_name="HL_vs_HL"
    )

    hypothesis_df.to_excel(
        writer,
        sheet_name="Hypothesis_Level",
        index=False
    )

    hypothesis_dim_df.to_excel(
        writer,
        sheet_name="Hypothesis_Level_By_Dimension",
        index=False
    )

    expert_df.to_excel(
        writer,
        sheet_name="Expert_Level",
        index=False
    )

    expert_cross_df.to_excel(
        writer,
        sheet_name="Expert_Cross_Level",
        index=False
    )


# ======================================
# Print the statistical results
# ======================================
print("\n===== Summary =====")
print(summary_df)

print("\n===== Cross-group HH vs HL =====")
print(cross_df)

print("\n===== HH vs HH =====")
print(hh_df)

print("\n===== HL vs HL =====")
print(hl_df)

print("\n===== Hypothesis-Level Stats =====")
print(hypothesis_df)

print("\n===== Hypothesis-Level Stats by Dimension =====")
print(hypothesis_dim_df)

print("\n===== Expert-Level Stats =====")
print(expert_df)

print("\n===== Expert Cross-Level Stats =====")
print(expert_cross_df)

print("\nSaved: Hypothesis_Analysis_Result.xlsx")



