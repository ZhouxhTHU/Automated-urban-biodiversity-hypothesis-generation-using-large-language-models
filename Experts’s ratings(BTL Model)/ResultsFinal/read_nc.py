"""
Verify all reported statistics from idata_Novelty.nc / idata_Significance.nc /
idata_Testability.nc, and export the results to an Excel workbook.

Requires: arviz, numpy, pandas, openpyxl
"""

import numpy as np
import pandas as pd
import arviz as az

FILES = {
    "Novelty": "/mnt/user-data/uploads/idata_Novelty.nc",
    "Significance": "/mnt/user-data/uploads/idata_Significance.nc",
    "Testability": "/mnt/user-data/uploads/idata_Testability.nc",
}


def eti(x, prob=0.95):
    """Equal-tailed interval: raw quantiles."""
    lo = np.quantile(x, (1 - prob) / 2)
    hi = np.quantile(x, 1 - (1 - prob) / 2)
    return lo, hi


def hdi(x, prob=0.95):
    """True highest-density interval: shortest interval containing `prob` mass."""
    x = np.sort(x.flatten())
    n = len(x)
    k = int(np.floor(prob * n))          # number of samples the interval must contain
    n_candidates = n - k
    widths = x[k:] - x[:n_candidates]
    j = np.argmin(widths)
    return x[j], x[j + k]


def tie_prob_from_logit(tie_logit_draws):
    """Davidson tie model: nu = exp(tie_logit); P(tie | equal strength) = nu / (2 + nu)."""
    nu = np.exp(tie_logit_draws)
    return nu / (2 + nu)


rows_effects = []      # mu_LLM / sigma_LLM / tie probability per dimension
rows_diag = []          # convergence diagnostics per dimension

for dim, path in FILES.items():
    idata = az.from_netcdf(path)

    # ---- population-level source effect (mu_LLM) ----
    mu = idata.posterior["mu_LLM"].values.flatten()
    mu_mean = mu.mean()
    eti_lo, eti_hi = eti(mu, 0.95)
    hdi_lo, hdi_hi = hdi(mu, 0.95)

    # ---- between-expert heterogeneity (sigma_LLM) ----
    sigma = idata.posterior["sigma_LLM"].values.flatten()
    sigma_mean = sigma.mean()
    sigma_hdi_lo, sigma_hdi_hi = hdi(sigma, 0.95)

    # ---- tie probability at equal latent preference ----
    tie_logit = idata.posterior["tie_logit"].values.flatten()
    p_tie_draws = tie_prob_from_logit(tie_logit)
    p_tie_mean = p_tie_draws.mean()
    p_tie_hdi_lo, p_tie_hdi_hi = hdi(p_tie_draws, 0.95)

    rows_effects.append({
        "Dimension": dim,
        "mu_LLM mean": round(mu_mean, 4),
        "mu_LLM 95% ETI low": round(eti_lo, 4),
        "mu_LLM 95% ETI high": round(eti_hi, 4),
        "mu_LLM 95% HDI low": round(hdi_lo, 4),
        "mu_LLM 95% HDI high": round(hdi_hi, 4),
        "sigma_LLM mean": round(sigma_mean, 4),
        "sigma_LLM 95% HDI low": round(sigma_hdi_lo, 4),
        "sigma_LLM 95% HDI high": round(sigma_hdi_hi, 4),
        "Tie prob mean (%)": round(p_tie_mean * 100, 2),
        "Tie prob 95% HDI low (%)": round(p_tie_hdi_lo * 100, 2),
        "Tie prob 95% HDI high (%)": round(p_tie_hdi_hi * 100, 2),
    })

    # ---- convergence diagnostics (across ALL variables in the model) ----
    summ_all = az.summary(idata, ci_prob=0.95, ci_kind="eti", round_to="none")
    n_div = int(idata.sample_stats["diverging"].values.sum())
    n_draws_total = idata.sample_stats["diverging"].values.size

    rows_diag.append({
        "Dimension": dim,
        "Divergences": n_div,
        "Total draws": n_draws_total,
        "Max R-hat (all params)": round(summ_all["r_hat"].max(), 4),
        "Min ESS bulk (all params)": round(summ_all["ess_bulk"].min(), 1),
        "Min ESS tail (all params)": round(summ_all["ess_tail"].min(), 1),
    })

df_effects = pd.DataFrame(rows_effects)
df_diag = pd.DataFrame(rows_diag)

# Overall (pooled across the three models) convergence line, matching the paper's phrasing
overall = pd.DataFrame([{
    "Metric": "Across all 3 models",
    "Total divergences": df_diag["Divergences"].sum(),
    "Max R-hat": df_diag["Max R-hat (all params)"].max(),
    "Min ESS bulk": df_diag["Min ESS bulk (all params)"].min(),
    "Min ESS tail": df_diag["Min ESS tail (all params)"].min(),
}])

with pd.ExcelWriter("/mnt/user-data/outputs/DBT_model_verification.xlsx", engine="openpyxl") as writer:
    df_effects.to_excel(writer, sheet_name="Effects & Tie Prob", index=False)
    df_diag.to_excel(writer, sheet_name="Convergence per model", index=False)
    overall.to_excel(writer, sheet_name="Convergence overall", index=False)

print("Done.")
print(df_effects.to_string(index=False))
print()
print(df_diag.to_string(index=False))
print()
print(overall.to_string(index=False))
