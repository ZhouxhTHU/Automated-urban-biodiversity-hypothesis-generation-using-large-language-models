"""
Bayesian Hierarchical Davidson-BTL for LLM vs. Human Expert Evaluation.
Author: Jun Yang, Google Gemini; Date: 27 June 2026.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

# ==========================================
# Configuration & Setup
# ==========================================
# Dynamically locate the script's directory and create a 'Results' subfolder
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication-grade MCMC settings
# 2000 tune / 2000 draws is the standard for hierarchical model stability
DRAWS = 2000
TUNE = 4000
CHAINS = 4
CORES = 4
TARGET_ACCEPT = 0.99
RANDOM_SEED = 42

RHAT_MAX = 1.01
ESS_MIN_PER_CHAIN = 400
ROPE_LO = -0.1
ROPE_HI = 0.1

# ==========================================
# 1. Real Data Ingestion & Preprocessing
# ==========================================
def load_and_preprocess_data(file_path):
    """Loads empirical data and maps it to the BTL utility structure."""
    print(f"Loading data from: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Could not find the dataset at: {file_path}")
        
    df = pd.read_excel(file_path)
    
    # 1. Standardize core columns
    df = df.rename(columns={
        'Expert_ID': 'expert_id',
        'Dimension': 'dimension'
    })
    
    # 2. Extract Hypothesis Types from the IDs (Assuming prefixes LLM_ and HUM_)
    df['hypothesis_A_type'] = df['Hypothesis_A_ID'].apply(
        lambda x: 'LLM' if str(x).strip().startswith('LLM') else 'Human'
    )
    df['hypothesis_B_type'] = df['Hypothesis_B_ID'].apply(
        lambda x: 'LLM' if str(x).strip().startswith('LLM') else 'Human'
    )
    
    # 3. Map the Result column to explicit outcomes
    def map_outcome(row):
        res = str(row['Result']).strip()
        if res == 'Tie':
            return 'Tie'
        elif res == str(row['Hypothesis_A_ID']).strip():
            return 'A_wins'
        elif res == str(row['Hypothesis_B_ID']).strip():
            return 'B_wins'
        else:
            raise ValueError(f"Invalid Result '{res}' found for pair {row['Hypothesis_A_ID']} vs {row['Hypothesis_B_ID']}")

    df['outcome'] = df.apply(map_outcome, axis=1)
    
    return df

# ==========================================
# 2. Bayesian Model Compilation
# ==========================================
def build_and_run_model(df, dimension):
    """Fits the Hierarchical Davidson-BTL using centered utilities."""
    print(f"\n" + "="*50)
    print(f"Fitting Model for Dimension: {dimension}")
    print("="*50)
    
    df_dim = df[df['dimension'] == dimension].copy()
    
    # Map types (Human=0, LLM=1)
    type_map = {'Human': 0, 'LLM': 1}
    df_dim['A_idx'] = df_dim['hypothesis_A_type'].map(type_map)
    df_dim['B_idx'] = df_dim['hypothesis_B_type'].map(type_map)
    
    # Map outcomes to match the utility stack: [-0.5*diff (B), tie_logit (Tie), 0.5*diff (A)]
    outcome_map = {'B_wins': 0, 'Tie': 1, 'A_wins': 2}
    df_dim['out_idx'] = df_dim['outcome'].map(outcome_map)
    
    # Map experts for hierarchical indexing
    expert_ids = np.sort(df_dim['expert_id'].unique())
    expert_map = {eid: i for i, eid in enumerate(expert_ids)}
    df_dim['expert_idx'] = df_dim['expert_id'].map(expert_map)
    
    coords = {"expert": expert_ids}
    
    with pm.Model(coords=coords) as model:
        # Population-level consensus (Global LLM advantage)
        mu_llm = pm.Normal('mu_LLM', mu=0, sigma=1.0)
        sigma_llm = pm.HalfNormal('sigma_LLM', sigma=1.0)
        
        # Expert-level deviations (Non-centered parameterization)
        z_expert = pm.Normal('z_expert', mu=0, sigma=1.0, dims="expert")
        lambda_llm_expert = pm.Deterministic('lambda_LLM_expert', mu_llm + z_expert * sigma_llm, dims="expert")
        
        # Global tie parameter (in logit space)
        tie_logit = pm.Normal('tie_logit', mu=0.0, sigma=1.0)
        
        # Extract indices for tensor operations
        A_idx = df_dim['A_idx'].values
        B_idx = df_dim['B_idx'].values
        expert_idx = df_dim['expert_idx'].values
        y_obs = df_dim['out_idx'].values
        
        # Ability assignment: If LLM (1), use expert's LLM bias. If Human (0), use 0.0 baseline.
        lambda_A = pt.switch(pt.eq(A_idx, 1), lambda_llm_expert[expert_idx], 0.0)
        lambda_B = pt.switch(pt.eq(B_idx, 1), lambda_llm_expert[expert_idx], 0.0)
        
        merit_diff = lambda_A - lambda_B
        
        # Broadcast the scalar tie_logit to a 1D vector matching merit_diff's length
        tie_logit_vec = pt.ones_like(merit_diff) * tie_logit
        
        # Centered Davidson Utilities
        utilities = pt.stack([-0.5 * merit_diff, tie_logit_vec, 0.5 * merit_diff], axis=1)
        pm.Categorical('likelihood', logit_p=utilities, observed=y_obs)
        
        # Run Sampler
        print(f"Starting MCMC Sampler for {dimension} (Draws: {DRAWS}, Tune: {TUNE})...")
        idata = pm.sample(
            draws=DRAWS, 
            tune=TUNE, 
            chains=CHAINS, 
            cores=CORES,
            target_accept=TARGET_ACCEPT, 
            random_seed=RANDOM_SEED,
            init="adapt_diag",   # <--- ADD THIS LINE (Removes the random jitter)
            return_inferencedata=True, 
            progressbar=True
        )
        
    return model, idata, expert_ids

# ==========================================
# 3. Diagnostics & Outputs
# ==========================================
def save_diagnostics(idata, dimension):
    """Evaluates convergence (R-hat, Divergences, ESS) and saves to CSV."""
    print(f"\n--- Diagnostics for {dimension} ---")
    
    # Save InferenceData object for safety
    idata.to_netcdf(OUTPUT_DIR / f"idata_{dimension}.nc")
    
    summary = az.summary(idata, var_names=['mu_LLM', 'sigma_LLM', 'tie_logit'])
    summary.to_csv(OUTPUT_DIR / f"summary_{dimension}.csv")
    
    n_div = int(idata.sample_stats["diverging"].sum().values)
    rhat_max = float(summary['r_hat'].max())
    
    print(f"Divergences: {n_div}")
    print(f"Max R-hat:   {rhat_max:.4f}")
    
    if n_div > 0:
        warnings.warn(f"WARNING: Found {n_div} divergences.")
    if rhat_max > RHAT_MAX:
        warnings.warn(f"WARNING: High R-hat ({rhat_max}) detected. Chains may not have converged.")
        
    return summary

def plot_expert_bias(idata, expert_ids, dimension):
    """Generates a forest plot of expert-specific LLM biases."""
    
    # Set the figure size using standard matplotlib
    plt.figure(figsize=(10, max(6, len(expert_ids) * 0.25)))
    
    # Call plot_forest without the deprecated kwargs
    az.plot_forest(
        idata, 
        var_names=['lambda_LLM_expert'], 
        combined=True
    )
    
    # Draw our custom threshold lines
    plt.axvline(0, color='red', linestyle='--', alpha=0.6, label='Human Baseline (Neutral)')
    plt.axvline(ROPE_HI, color='gray', linestyle=':', alpha=0.5, label='ROPE Boundary')
    plt.axvline(ROPE_LO, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(f'Individual Expert Bias toward LLMs ({dimension})', fontsize=14)
    plt.xlabel('Latent Bias [ > 0 favors LLM, < 0 favors Human ]')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / f"expert_bias_forest_{dimension}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    
    dataset_path = SCRIPT_DIR / "All_Comparison_Results.xlsx"
    
    # 1. Load Data
    df = load_and_preprocess_data(dataset_path)
    print(f"Successfully loaded {len(df)} evaluations.")
    
    # 2. Iterate through every evaluation dimension found in the dataset
    dimensions = df['dimension'].unique()
    print(f"Found {len(dimensions)} dimensions to process: {', '.join(dimensions)}")
    
    for target_dimension in dimensions:
        
        # 3. Compile and Sample
        model, idata, experts = build_and_run_model(df, target_dimension)
        
        # 4. Check Convergence & Save Checkpoints
        summary = save_diagnostics(idata, target_dimension)
        print("\nPopulation Summary:")
        print(summary)
        
        # 5. Plot Results
        plot_expert_bias(idata, experts, target_dimension)
        
    print(f"\nAll modeling complete! Check the '{OUTPUT_DIR.name}' folder for all saved artifacts.")