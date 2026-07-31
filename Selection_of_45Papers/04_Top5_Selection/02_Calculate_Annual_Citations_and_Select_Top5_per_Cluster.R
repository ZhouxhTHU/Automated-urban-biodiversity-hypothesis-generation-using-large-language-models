# Calculate annual citation rates and select the five highest-ranked papers
# from each of the nine semantic clusters.
# The reference year is fixed at 2026 to reproduce the archived results.

library(dplyr)
library(readxl)
library(writexl)

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
if (length(script_arg) != 1) {
  stop("Unable to determine the script location. Run this file with Rscript.")
}

script_path <- normalizePath(sub("^--file=", "", script_arg))
bundle_root <- dirname(dirname(script_path))

input_path <- file.path(
  bundle_root,
  "03_Clustering",
  "1487_Papers_KMedoids_9_Clusters.xlsx"
)
full_output_path <- file.path(
  dirname(script_path),
  "1487_Papers_9_Clusters_and_Annual_Citations_2026_Reproduced.xlsx"
)
selected_output_path <- file.path(
  dirname(script_path),
  "Final_45_Papers_Top5_per_Cluster_Reproduced.csv"
)

reference_year <- 2026L
expected_clusters <- 0:8

df <- read_xlsx(input_path)
required_columns <- c(
  "rec_index",
  "txt_title",
  "txt_year",
  "Times Cited, WoS Core",
  "Abstract",
  "cluster_id"
)
missing_columns <- setdiff(required_columns, names(df))
if (length(missing_columns) > 0) {
  stop(paste("Missing input columns:", paste(missing_columns, collapse = ", ")))
}
if (nrow(df) != 1487) {
  stop(paste("Expected 1,487 input rows; found", nrow(df)))
}

df <- df %>%
  mutate(
    TCperYear = `Times Cited, WoS Core` /
      (reference_year - as.integer(txt_year) + 1L)
  )

cluster_ids <- sort(unique(df$cluster_id))
if (!identical(as.integer(cluster_ids), expected_clusters)) {
  stop(paste("Incomplete cluster IDs:", paste(cluster_ids, collapse = ", ")))
}

final_45 <- df %>%
  filter(!is.na(cluster_id)) %>%
  group_by(cluster_id) %>%
  arrange(desc(TCperYear), .by_group = TRUE) %>%
  slice_head(n = 5) %>%
  ungroup() %>%
  arrange(cluster_id)

selected_counts <- table(final_45$cluster_id)
if (nrow(final_45) != 45 || any(selected_counts != 5)) {
  stop("The final result is not nine clusters with five papers per cluster.")
}

write_xlsx(df, full_output_path)
write.csv(
  final_45[, c(
    "cluster_id",
    "txt_title",
    "txt_year",
    "Times Cited, WoS Core",
    "TCperYear",
    "Abstract"
  )],
  selected_output_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("Full reproduced results:", full_output_path, "\n")
cat("Final 45 reproduced papers:", selected_output_path, "\n")
