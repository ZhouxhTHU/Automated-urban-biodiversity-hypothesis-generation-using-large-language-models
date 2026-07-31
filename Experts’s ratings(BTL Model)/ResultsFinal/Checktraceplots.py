"""
Script for Checking Trace Plots.
Author: Jun Yang, Google Gemini; Date: 27 June 2026.
"""

from pathlib import Path
import arviz as az
import matplotlib.pyplot as plt

# Dynamically define the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Set the path to the Excel file (one folder up)
dataset_path = SCRIPT_DIR.parent / "All_Comparison_Results.xlsx"

# Load a dimension that actually exists in your data
file_path = SCRIPT_DIR / "idata_Novelty.nc"
print(f"Loading {file_path}...")
idata = az.from_netcdf(file_path)

# 1. Check the raw unrounded float values
raw_rhat = az.rhat(idata)
print("\n--- Raw Unrounded R-hat ---")
# REMOVED .posterior from these two lines:
print("mu_LLM:    ", float(raw_rhat['mu_LLM'].values))
print("sigma_LLM: ", float(raw_rhat['sigma_LLM'].values))

# 2. Plot the traces (The "fuzzy caterpillars")
print("\nGenerating Trace Plot...")
az.plot_trace(idata, var_names=['mu_LLM', 'sigma_LLM', 'tie_logit'])
plt.tight_layout()
plt.show()

# 3. Plot the ranks
print("Generating Rank Plot...")
az.plot_rank(idata, var_names=['mu_LLM', 'sigma_LLM', 'tie_logit'])
plt.tight_layout()
plt.show()