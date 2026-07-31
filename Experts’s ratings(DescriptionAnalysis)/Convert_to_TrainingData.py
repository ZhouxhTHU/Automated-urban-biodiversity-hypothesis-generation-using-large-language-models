import pandas as pd
import glob
import os


# ======================================
# Configuration
# ======================================
assignment_file = "All_Assignment_Pairs.xlsx"
survey_pattern = os.path.join(
    "Expert_Results",
    "*Urban Biodiversity Hypothesis Project.xlsx"
)


# ======================================
# ID conversion function: HH9 -> HUM_9, HL9 -> LLM_9
# ======================================
def convert_id(raw_id: str) -> str:
    raw_id = str(raw_id).strip()
    if raw_id.upper().startswith("HH"):
        num = raw_id[2:]
        return f"HUM_{num}"
    elif raw_id.upper().startswith("HL"):
        num = raw_id[2:]
        return f"LLM_{num}"
    else:
        return raw_id  # Leave unrecognized IDs unchanged


# ======================================
# Load the assignment table
# ======================================
assignment_df = pd.read_excel(assignment_file)
assignment_df = assignment_df.sort_values(
    ["Expert_ID", "Pair_Number"]
).reset_index(drop=True)

dimensions = ["Novelty", "Significance", "Testability"]

# ======================================
# Load the survey files
# ======================================
survey_files = glob.glob(survey_pattern)
print(f"Found {len(survey_files)} survey files")

rows = []

for survey_file in survey_files:

    expert_id = os.path.basename(survey_file).split()[0]

    expert_pairs = assignment_df[
        assignment_df["Expert_ID"] == expert_id
    ].sort_values("Pair_Number")

    if len(expert_pairs) == 0:
        print(f"  [SKIP] {expert_id}: not found in assignment table")
        continue

    survey_df = pd.read_excel(survey_file)

    # The second row (index=1) contains the actual responses
    response_row = survey_df.iloc[1]

    question_cols = [
        col for col in survey_df.columns
        if str(col).startswith("Which hypothesis")
    ]

    expected_num = len(expert_pairs) * 3

    if len(question_cols) < expected_num:
        print(f"  [SKIP] {expert_id}: expected {expected_num} cols, got {len(question_cols)}")
        continue

    print(f"  [OK] {expert_id}: {len(expert_pairs)} pairs")

    for i, (_, pair) in enumerate(expert_pairs.iterrows()):

        hypA_raw = pair["Hypothesis_A_ID"]
        hypB_raw = pair["Hypothesis_B_ID"]

        hypA = convert_id(hypA_raw)
        hypB = convert_id(hypB_raw)

        pair_cols = question_cols[i * 3:(i + 1) * 3]

        for j, col in enumerate(pair_cols):

            dimension = dimensions[j]
            answer = response_row[col]

            if pd.isna(answer):
                result = "NA"

            elif "Tie" in str(answer):
                result = "Tie"

            elif str(answer).strip() == "Hypothesis A":
                result = hypA

            elif str(answer).strip() == "Hypothesis B":
                result = hypB

            else:
                result = f"UNKNOWN({answer})"

            rows.append({
                "Expert_ID":       expert_id,
                "Hypothesis_A_ID": hypA,
                "Hypothesis_B_ID": hypB,
                "Dimension":       dimension,
                "Result":          result,
            })

# ======================================
# Save the results
# ======================================
result_df = pd.DataFrame(rows, columns=[
    "Expert_ID", "Hypothesis_A_ID", "Hypothesis_B_ID", "Dimension", "Result"
])

output_file = "All_Comparison_Results.xlsx"
result_df.to_excel(output_file, index=False)

print(f"\nTotal rows: {len(result_df)}")
print(f"Saved: {output_file}")
print("\nPreview:")
print(result_df.head(9).to_string(index=False))
