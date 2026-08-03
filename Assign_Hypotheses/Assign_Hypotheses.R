library(openxlsx)
library(igraph)
library(dplyr)

set.seed(42)

# =============================================================================
# Parameter settings
# =============================================================================
N_EXPERTS        <- 30L   # Number of experts
ITEMS_PER_EXPERT <- 45L   # Total hypotheses assigned to each expert
PAIRS_PER_EXPERT <- 45L   # Circular chain: 45 nodes → 45 edges
N_HH             <- 45L   # Number of human-generated hypotheses
N_HL             <- 45L   # Number of LLM-generated hypotheses
TARGET_INTRA_PCT <- 0.20  # Target proportion of within-group pairs
TARGET_INTRA_N   <- round(PAIRS_PER_EXPERT * TARGET_INTRA_PCT)  # 9 within-group pairs per expert
MAX_TRIES        <- 10000L

cat(strrep("=", 80), "\n")
cat("Hypothesis assignment script (HH/HL groups, 20% within-group pairs, circular chain)\n")
cat(strrep("=", 80), "\n\n")

# =============================================================================
# Step 1: Load data
# =============================================================================
cat("Step 1: Loading data...\n")

# Read the hypothesis file (columns: Hypothesis_ID and Hypothesis_content)
hyp_df <- read.xlsx("90Hypotheses.xlsx")  # All 90 hypotheses
names(hyp_df) <- trimws(names(hyp_df))

# Split into groups
hh_df <- hyp_df %>% filter(grepl("^HH", Hypothesis_ID)) %>% arrange(Hypothesis_ID)
hl_df <- hyp_df %>% filter(grepl("^HL", Hypothesis_ID)) %>% arrange(Hypothesis_ID)

all_hh <- hh_df$Hypothesis_ID  # HH01~HH45
all_hl <- hl_df$Hypothesis_ID  # HL01~HL45
all_ids <- c(all_hh, all_hl)

all_content <- setNames(hyp_df$Hypothesis_content, hyp_df$Hypothesis_ID)

cat(sprintf("  Human-generated hypotheses (HH): %d\n", length(all_hh)))
cat(sprintf("  LLM-generated hypotheses   (HL): %d\n", length(all_hl)))

# Read the expert exclusion-constraint file (columns: Expert_ID and HH_Hypothesis)
# Separate multiple IDs in HH_Hypothesis with commas, e.g. "HH01, HH03, HH05"
excl_df <- read.xlsx("Expert_Exclusions.xlsx")
names(excl_df) <- trimws(names(excl_df))

# Build an exclusion list named by Expert_ID; each value is a vector of ineligible HH IDs
expert_ids <- excl_df$Expert_ID
excl_list  <- lapply(seq_len(nrow(excl_df)), function(i) {
  raw <- as.character(excl_df$HH_Hypothesis[i])
  if (is.na(raw) || raw == "") return(character(0))
  trimws(unlist(strsplit(raw, ",")))
})
names(excl_list) <- expert_ids

cat(sprintf("  Number of experts: %d\n", length(expert_ids)))
cat(sprintf("  Experts with exclusion constraints: %d\n",
            sum(sapply(excl_list, length) > 0)))
cat("\n")

# =============================================================================
# Step 2: Build the HH eligibility matrix
# =============================================================================
cat("Step 2: Building the HH eligibility matrix...\n")

eligible_hh <- matrix(
  TRUE,
  nrow = length(expert_ids),
  ncol = length(all_hh),
  dimnames = list(expert_ids, all_hh)
)

for (exp in expert_ids) {
  bad <- excl_list[[exp]]
  bad <- intersect(bad, all_hh)
  if (length(bad) > 0) eligible_hh[exp, bad] <- FALSE
}

avail_hh <- rowSums(eligible_hh)
cat(sprintf("  Eligible HH hypotheses per expert: mean=%.1f, range=[%d, %d]\n",
            mean(avail_hh), min(avail_hh), max(avail_hh)))

# Check whether enough HH hypotheses are eligible (each expert needs about 22–23 HH hypotheses)
# Actual HH count per expert: 30*45/90*45 = 22.5, so some experts receive 23 and others 22
# Exact allocation: assign 23 HH to half of the experts and 22 to the other half, giving 45*30 = 1,350 HH appearances
# Each HH should therefore be seen 1,350/45 = 30 times
N_HH_PER_EXPERT <- N_HH  # Initially set to 45; balance by group below
# Each expert receives 45 hypotheses. HH and HL counts must make every hypothesis appear 45*2*30/90 = 30 times
# Total HH exposures = n_hh_per_expert * 30 = 45*30 = 1,350 → n_hh_per_expert = 1,350/30 = 45?
# Incorrect: 45 hypotheses per expert = n_hh_i + n_hl_i; total HH exposure = sum(n_hh_i), which should equal 45*30 = 1,350
# Yet the number of views per HH = total HH exposure/45 = 1,350/45 = 30 ✓
# Likewise, total HL exposure = 1,350, so each HL is seen 30 times
# Thus sum(n_hh_i) = 1,350 and sum(n_hl_i) = 1,350 across experts
# If equal for everyone: n_hh_i = 1,350/30 = 45? Incorrect, because the total number of hypotheses per expert is only 45
# Correct interpretation: of the 45 hypotheses assigned to each expert, the HH count is h_i and the HL count is 45 - h_i
# Total HH exposure = sum(h_i) = 45*30 = 1,350, so mean(h_i) = 1,350/30 = 45?
# This would mean assigning all 45 HH hypotheses to every expert, leaving no room for HL hypotheses. Incorrect
# ---- Derive again ----
# Target: each hypothesis is "compared" 30 times (appears in 30 different pairs)
# Each expert receives 45 pairs. In a circular chain, every hypothesis appears exactly twice
# Therefore, experts assigned each hypothesis × 2 = comparisons per hypothesis → experts assigned = 15
# Also, 45 hypotheses * 15 experts = 675 = 30 experts * 45/2 → 675 ✓
# Thus each expert receives 45 hypotheses, and each hypothesis must be assigned to 675/45 = 15 experts
# Across 30 experts, there are 30*45 = 1,350 assignments (15 assignments per hypothesis)
# If the HH count per expert is h_i, sum(h_i) = 45*15 = 675 → mean(h_i) = 675/30 = 22.5
# Therefore, 15 experts receive 23 HH hypotheses and 15 receive 22 HH hypotheses (or an equivalent allocation)
# Likewise for HL: 45*15 = 675, mean = 22.5

# Verify that the eligible count is at least 23
if (any(avail_hh < 23)) {
  cat("  Warning: the following experts have fewer than 23 eligible HH hypotheses; assignment will be best-effort:\n")
  cat(paste("   ", expert_ids[avail_hh < 23], collapse=", "), "\n")
}

# Determine the HH count per expert (22 or 23, summing to 675)
# One option is to sort by exclusions in descending order and assign fewer to experts with more exclusions (but at least 22)
# Simple strategy: randomly assign 23 HH hypotheses to 15 experts and 22 to the rest
set.seed(42)
n_hh_per <- rep(22L, N_EXPERTS)
n_hh_per[sample(N_EXPERTS, 15)] <- 23L
n_hl_per <- ITEMS_PER_EXPERT - n_hh_per  # 22 or 23

cat(sprintf("\n  HH assignment: %d experts receive 22, %d receive 23 (total=%d, each HH assigned %d times)\n",
            sum(n_hh_per==22), sum(n_hh_per==23), sum(n_hh_per), sum(n_hh_per)/N_HH))
cat(sprintf("  HL assignment: %d experts receive 22, %d receive 23 (total=%d, each HL assigned %d times)\n\n",
            sum(n_hl_per==22), sum(n_hl_per==23), sum(n_hl_per), sum(n_hl_per)/N_HL))

# =============================================================================
# Step 3 (revised): Strictly balanced assignment
# =============================================================================
cat("Step 3: Running strictly balanced assignment (each hypothesis must be selected 15 times)...\n")

# Define a general strictly balanced assignment function
assign_strictly_balanced <- function(all_ids, n_per_expert, total_slots, expert_ids, eligible_matrix = NULL) {
  # total_slots = 675 (45 hypotheses * 15 assignments)
  # n_per_expert = vector giving 22 or 23 hypotheses per expert

  for (attempt in 1:20000) {
    # Track the remaining assignment quota for each ID (initially 15 per ID)
    remaining_quota <- setNames(rep(15L, length(all_ids)), all_ids)
    assignment <- lapply(expert_ids, function(x) character(0))
    names(assignment) <- expert_ids

    # Process experts in randomized order
    shuffled_experts <- sample(seq_along(expert_ids))
    success <- TRUE

    for (i in shuffled_experts) {
      exp_id <- expert_ids[i]
      needed <- n_per_expert[i]

      # Find hypotheses with remaining quota that are not excluded for this expert
      if (!is.null(eligible_matrix)) {
        eligible_for_this_exp <- names(which(eligible_matrix[exp_id, ]))
        pool <- intersect(eligible_for_this_exp, names(remaining_quota[remaining_quota > 0]))
      } else {
        pool <- names(remaining_quota[remaining_quota > 0])
      }

      if (length(pool) < needed) {
        success <- FALSE
        break
      }

      # Prioritize hypotheses with the largest remaining quota to improve stability
      pool_weights <- remaining_quota[pool]
      # Add a small random perturbation to avoid selecting the same set every time
      chosen <- sample(pool, size = needed, prob = pool_weights + runif(length(pool), 0, 0.5))

      assignment[[exp_id]] <- chosen
      remaining_quota[chosen] <- remaining_quota[chosen] - 1L
    }

    if (success) {
      cat(sprintf("  ✓ Found a strictly balanced solution on attempt %d\n", attempt))
      return(assignment)
    }
  }
  stop("A strictly balanced assignment is infeasible under the current constraints. Check whether the exclusions are too restrictive.")
}

# 1. Strictly assign HH hypotheses
res_hh_list <- assign_strictly_balanced(all_hh, n_hh_per, 675, expert_ids, eligible_hh)
assign_hh <- res_hh_list

# 2. Strictly assign HL hypotheses (easier because there are no exclusion constraints)
res_hl_list <- assign_strictly_balanced(all_hl, n_hl_per, 675, expert_ids, NULL)
assign_hl <- res_hl_list

# =============================================================================
# Step 4: Circular-chain pairing (control the within-group proportion at about 20%)
#
# Core principle (using n_hh=22 and n_hl=23 as an example):
#   Arrange the hypotheses in a 45-node cycle, giving 45 edges between adjacent nodes.
#   Each edge is either "cross-group" (HH-HL) or "within-group" (HH-HH or HL-HL).
#
#   Construction method—the alternating-block method:
#     1. Divide the n_hh HH hypotheses into blocks of length 1 or 2
#        - A length-2 block produces one HH-HH within-group edge
#        - If the target number of HH within-group edges is intra_hh, use intra_hh double blocks and
#          (n_hh - 2*intra_hh) single blocks, for a total of (n_hh - intra_hh) HH blocks
#     2. Similarly, create intra_hl HL within-group edges and (n_hl - intra_hl) HL blocks
#     3. Arrange all blocks in a strictly alternating HH-block → HL-block → HH-block → ... chain
#        (randomizing block order within each group); every seam is an HH-HL cross-group edge
#     4. Total edges = 45, by the circular-chain property
#        Within-group edges = intra_hh + intra_hl (inside double blocks)
#        Cross-group edges = total number of blocks (at seams between blocks)
#             = (n_hh - intra_hh) + (n_hl - intra_hl)
#             = 45 - target_intra ✓
#
#   Notes:
#     - Strict alternation requires the group block counts to differ by no more than 1;
#       otherwise, adjacent same-group blocks at the end would add a within-group edge
#       Since n_hh_blk = n_hh - intra_hh and n_hl_blk = n_hl - intra_hl,
#       their difference is about 1 and satisfies this condition
#     - The final and first blocks are also connected: final block (group A) → first block (group B)
#       is a cross-group edge already included in the block count
#       If the final and first blocks belong to the same group, an extra within-group edge is created
#       and must be detected and corrected
# =============================================================================
cat("\nStep 4: Generating circular-chain pairs (target ≈9 within-group pairs)...\n")

make_chain_pairs <- function(hh_vec, hl_vec, target_intra = TARGET_INTRA_N) {
  n_hh <- length(hh_vec)
  n_hl <- length(hl_vec)

  # Allocate target within-group edges in proportion to group size
  intra_hh <- round(target_intra * n_hh / (n_hh + n_hl))
  intra_hl <- target_intra - intra_hh
  intra_hh <- max(0L, min(intra_hh, n_hh %/% 2L))
  intra_hl <- max(0L, min(intra_hl, n_hl %/% 2L))

  hh_s <- sample(hh_vec)
  hl_s <- sample(hl_vec)

  # Build the block list: each block has length 1 (single) or 2 (double, contributing one within-group edge)
  # Place intra_hh double blocks first; all remaining blocks are single
  build_blocks <- function(items, n_double) {
    blocks <- vector("list", length(items) - n_double)
    for (k in seq_len(n_double)) {
      blocks[[k]] <- items[c(2*k - 1, 2*k)]
    }
    singles <- items[(2*n_double + 1):length(items)]
    for (j in seq_along(singles)) {
      blocks[[n_double + j]] <- singles[j]
    }
    # Shuffle block order to preserve randomness
    blocks[sample(length(blocks))]
  }

  hh_blocks <- build_blocks(hh_s, intra_hh)  # n_hh - intra_hh blocks
  hl_blocks <- build_blocks(hl_s, intra_hl)  # n_hl - intra_hl blocks

  n_hh_blk <- length(hh_blocks)
  n_hl_blk <- length(hl_blocks)

  # Concatenate in strict alternation: HH block, HL block, HH block, HL block, ...
  # The block counts differ by at most 1; append the remaining block from the larger group
  chain <- c()
  for (k in seq_len(min(n_hh_blk, n_hl_blk))) {
    chain <- c(chain, hh_blocks[[k]], hl_blocks[[k]])
  }
  # Append extra blocks when the block counts differ
  if (n_hh_blk > n_hl_blk) {
    for (k in (n_hl_blk + 1):n_hh_blk) chain <- c(chain, hh_blocks[[k]])
  } else if (n_hl_blk > n_hh_blk) {
    for (k in (n_hh_blk + 1):n_hl_blk) chain <- c(chain, hl_blocks[[k]])
  }

  # Generate circular-chain pairs, connecting the last item back to the first
  item_left  <- chain
  item_right <- c(chain[-1], chain[1])
  data.frame(
    Hypothesis_A_ID = item_left,
    Hypothesis_B_ID = item_right,
    stringsAsFactors = FALSE
  )
}

# Generate pairs for each expert
all_pairs_list <- vector("list", N_EXPERTS)

for (i in seq_along(expert_ids)) {
  exp    <- expert_ids[i]
  hh_vec <- assign_hh[[exp]]
  hl_vec <- assign_hl[[exp]]

  pairs <- make_chain_pairs(hh_vec, hl_vec, TARGET_INTRA_N)

  all_pairs_list[[i]] <- data.frame(
    Expert_ID       = exp,
    Pair_Number     = seq_len(nrow(pairs)),
    pairs,
    stringsAsFactors = FALSE
  )
}

all_pairs <- bind_rows(all_pairs_list)

# Add hypothesis content
all_pairs <- all_pairs %>%
  mutate(
    Hypothesis_A_content = all_content[Hypothesis_A_ID],
    Hypothesis_B_content = all_content[Hypothesis_B_ID]
  )

cat(sprintf("  Generated %d pairs (%d experts × %d pairs)\n\n",
            nrow(all_pairs), N_EXPERTS, PAIRS_PER_EXPERT))

# =============================================================================
# Step 5: Validate
# =============================================================================
cat("Step 5: Validating design constraints...\n")

# Group mapping
group_map <- c(
  setNames(rep("HH", N_HH), all_hh),
  setNames(rep("HL", N_HL), all_hl)
)

# 5.1 Exclusion-constraint check
violations <- 0L
for (exp in expert_ids) {
  bad_hh <- excl_list[[exp]]
  assigned_hh <- assign_hh[[exp]]
  overlap <- intersect(assigned_hh, bad_hh)
  if (length(overlap) > 0) {
    cat(sprintf("  ✗ Violation: %s was assigned excluded hypotheses: %s\n", exp, paste(overlap, collapse=", ")))
    violations <- violations + 1L
  }
}
cat(sprintf("  [5.1] Exclusion-constraint violations: %d %s\n",
            violations, ifelse(violations == 0, "✓", "✗")))

# 5.2 Number assigned to each expert
count_ok <- all(sapply(expert_ids, function(exp) {
  length(assign_hh[[exp]]) + length(assign_hl[[exp]]) == ITEMS_PER_EXPERT
}))
cat(sprintf("  [5.2] Every expert receives exactly %d hypotheses: %s\n",
            ITEMS_PER_EXPERT, ifelse(count_ok, "✓", "✗")))

# 5.3 Assignments per hypothesis
hh_assign_count <- table(unlist(assign_hh))
hl_assign_count <- table(unlist(assign_hl))
cat(sprintf("  [5.3] HH assignment count: mean=%.1f, range=[%d,%d]\n",
            mean(hh_assign_count), min(hh_assign_count), max(hh_assign_count)))
cat(sprintf("        HL assignment count: mean=%.1f, range=[%d,%d]\n",
            mean(hl_assign_count), min(hl_assign_count), max(hl_assign_count)))

# 5.4 Algebraic connectivity of the full graph
full_pairs_df <- all_pairs %>% select(Hypothesis_A_ID, Hypothesis_B_ID)
g_full  <- graph_from_data_frame(full_pairs_df, directed = FALSE, vertices = all_ids)
L_full  <- as.matrix(laplacian_matrix(g_full, normalized = FALSE))
eigs    <- sort(eigen(L_full, symmetric = TRUE, only.values = TRUE)$values)
lambda2 <- eigs[2]
n_comp  <- components(g_full)$no
cat(sprintf("  [5.4] Full-graph algebraic connectivity λ₂=%.4f (%s), connected components=%d\n",
            lambda2, ifelse(lambda2 > 0, "✓ connected", "✗ disconnected"), n_comp))

# 5.5 Comparisons per hypothesis (appearances in pairs)
hyp_cmp_count <- table(c(all_pairs$Hypothesis_A_ID, all_pairs$Hypothesis_B_ID))
cat(sprintf("  [5.5] Comparisons per hypothesis: mean=%.1f, range=[%d,%d] %s\n",
            mean(hyp_cmp_count), min(hyp_cmp_count), max(hyp_cmp_count),
            ifelse(max(hyp_cmp_count)-min(hyp_cmp_count)<=4,"✓","⚠")))

# 5.6 Within-group and cross-group pair statistics
pair_types <- all_pairs %>%
  mutate(
    grp_a = group_map[Hypothesis_A_ID],
    grp_b = group_map[Hypothesis_B_ID],
    type  = ifelse(grp_a == grp_b,
                   paste0("intra-", grp_a),
                   "inter-HH-HL")
  )

type_summary <- pair_types %>% group_by(type) %>% summarise(n=n(), .groups="drop")
total_pairs  <- nrow(all_pairs)
intra_n      <- sum(pair_types$grp_a == pair_types$grp_b)
inter_n      <- total_pairs - intra_n
intra_pct    <- intra_n / total_pairs

cat(sprintf("  [5.6] Within-group pairs: %d (%.1f%%), cross-group pairs: %d (%.1f%%) %s\n",
            intra_n, intra_pct*100,
            inter_n, (1-intra_pct)*100,
            ifelse(abs(intra_pct - TARGET_INTRA_PCT) < 0.05, "✓", "⚠")))
print(type_summary)

# 5.7 Within-group pairs per expert
intra_per_expert <- pair_types %>%
  group_by(Expert_ID) %>%
  summarise(intra = sum(grp_a == grp_b), .groups = "drop")
cat(sprintf("  [5.7] Within-group pairs per expert: mean=%.1f, range=[%d,%d]\n",
            mean(intra_per_expert$intra),
            min(intra_per_expert$intra),
            max(intra_per_expert$intra)))

# 5.8 Each hypothesis appears exactly twice per expert (circular-chain property)
chain_ok <- all(sapply(expert_ids, function(exp) {
  ep  <- all_pairs %>% filter(Expert_ID == exp)
  tbl <- table(c(ep$Hypothesis_A_ID, ep$Hypothesis_B_ID))
  all(tbl == 2)
}))
cat(sprintf("  [5.8] Circular chain: each hypothesis appears exactly twice per expert: %s\n\n",
            ifelse(chain_ok, "✓", "✗")))

# =============================================================================
# Step 6: Export files
# =============================================================================
cat("Step 6: Saving output files...\n")

# 6.1 Complete pair table
output_pairs <- all_pairs %>%
  select(Expert_ID, Pair_Number,
         Hypothesis_A_ID, Hypothesis_A_content,
         Hypothesis_B_ID, Hypothesis_B_content)

write.xlsx(output_pairs, "All_Assignment_Pairs.xlsx", overwrite = TRUE)
cat("  ✓ All_Assignment_Pairs.xlsx\n")

# 6.2 Questionnaire with a separate worksheet for each expert
wb_quest <- createWorkbook()

for (exp in expert_ids) {
  ep <- all_pairs %>%
    filter(Expert_ID == exp) %>%
    select(Pair_Number,
           Hypothesis_A_ID, Hypothesis_A_content,
           Hypothesis_B_ID, Hypothesis_B_content) %>%
    .[sample(nrow(.)), ]  # Randomize row order
  ep$Pair_Number <- seq_len(nrow(ep))

  addWorksheet(wb_quest, exp)
  writeData(wb_quest, exp, ep)
}

saveWorkbook(wb_quest, "Expert_Questionnaire.xlsx", overwrite = TRUE)
cat("  ✓ Expert_Questionnaire.xlsx\n")

# 6.2b Hypothesis assignment table with a separate worksheet for each expert
wb_assign <- createWorkbook()

for (exp in expert_ids) {
  assigned_ids <- c(assign_hh[[exp]], assign_hl[[exp]])
  exp_df <- data.frame(
    Hypothesis_ID      = assigned_ids,
    Hypothesis_content = all_content[assigned_ids],
    stringsAsFactors   = FALSE
  )
  addWorksheet(wb_assign, exp)
  writeData(wb_assign, exp, exp_df)
}

saveWorkbook(wb_assign, "Expert_Hypothesis_Assignment.xlsx", overwrite = TRUE)
cat("  ✓ Expert_Hypothesis_Assignment.xlsx\n")

# 6.3 Summary statistics
all_pass <- violations == 0 && lambda2 > 0 && count_ok && chain_ok &&
  abs(intra_pct - TARGET_INTRA_PCT) < 0.05

summary_df <- data.frame(
  Metric = c(
    "Total hypotheses", "Human-generated hypotheses (HH)", "LLM-generated hypotheses (HL)",
    "Number of experts", "Hypotheses per expert", "Pairs per expert", "Total pairs",
    "Exclusion-constraint violations",
    "Full-graph algebraic connectivity (λ₂)", "Connected components in full graph",
    "Comparisons per hypothesis (mean)", "Comparisons per hypothesis (range)",
    "Within-group pairs", "Within-group pair share (%)", "Cross-group pairs",
    "Within-group pairs per expert (mean)",
    "Each hypothesis appears twice in the circular chain",
    "Overall validation"
  ),
  Value = c(
    length(all_ids), N_HH, N_HL,
    N_EXPERTS, ITEMS_PER_EXPERT, PAIRS_PER_EXPERT, nrow(all_pairs),
    violations,
    round(lambda2, 4), n_comp,
    round(mean(hyp_cmp_count), 1),
    paste0("[", min(hyp_cmp_count), ", ", max(hyp_cmp_count), "]"),
    intra_n, round(intra_pct * 100, 1), inter_n,
    round(mean(intra_per_expert$intra), 1),
    ifelse(chain_ok, "Yes", "No"),
    ifelse(all_pass, "All checks passed ✓", "Issues detected; please review")
  ),
  stringsAsFactors = FALSE
)

write.xlsx(summary_df, "Summary_Statistics.xlsx", overwrite = TRUE)
cat("  ✓ Summary_Statistics.xlsx\n")

# =============================================================================
# Final report
# =============================================================================
cat("\n")
cat(strrep("=", 80), "\n")
if (all_pass) {
  cat("✓ All checks passed. The assignment design was completed successfully.\n")
} else {
  cat("⚠️  Issues were detected. Review the items marked ✗ or ⚠ above.\n")
}
cat(strrep("=", 80), "\n\n")
cat("Output files:\n")
cat("  1. All_Assignment_Pairs.xlsx   — All 1,350 pairs\n")
cat("  2. Expert_Questionnaire.xlsx   — Separate questionnaire worksheet for each expert\n")
cat("  3. Summary_Statistics.xlsx     — Summary statistics and validation\n")
cat("  4. Expert_Hypothesis_Assignment.xlsx — Hypotheses assigned to each expert\n")
