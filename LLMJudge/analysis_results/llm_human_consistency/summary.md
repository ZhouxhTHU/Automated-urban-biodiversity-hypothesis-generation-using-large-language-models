# LLM-Human Consistency Summary

## Overall

| model | n | exact | kappa | AC1 | AC2_quad | human tie | LLM tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude | 4050 | 0.405 | 0.010 | 0.159 | 0.014 | 0.200 | 0.004 |
| gemini | 4050 | 0.414 | 0.028 | 0.169 | 0.032 | 0.200 | 0.015 |
| gpt | 4050 | 0.410 | 0.017 | 0.166 | 0.021 | 0.200 | 0.000 |

## By Dimension

| model | dimension | n | exact | kappa | AC1 | human tie | LLM tie |
|---|---|---:|---:|---:|---:|---:|---:|
| claude | Novelty | 1350 | 0.430 | 0.025 | 0.200 | 0.164 | 0.011 |
| claude | Significance | 1350 | 0.394 | -0.009 | 0.143 | 0.201 | 0.000 |
| claude | Testability | 1350 | 0.392 | 0.016 | 0.134 | 0.233 | 0.000 |
| gemini | Novelty | 1350 | 0.435 | 0.032 | 0.207 | 0.164 | 0.010 |
| gemini | Significance | 1350 | 0.410 | 0.019 | 0.166 | 0.201 | 0.001 |
| gemini | Testability | 1350 | 0.398 | 0.030 | 0.136 | 0.233 | 0.034 |
| gpt | Novelty | 1350 | 0.439 | 0.036 | 0.213 | 0.164 | 0.001 |
| gpt | Significance | 1350 | 0.401 | 0.003 | 0.155 | 0.201 | 0.000 |
| gpt | Testability | 1350 | 0.390 | 0.011 | 0.131 | 0.233 | 0.000 |

## Aggregate Score Correlations

| model | dimension | hypotheses | pearson | spearman | mean diff |
|---|---|---:|---:|---:|---:|
| claude | Novelty | 90 | 0.185 | 0.157 | -0.000 |
| claude | Significance | 90 | -0.044 | -0.040 | -0.000 |
| claude | Testability | 90 | 0.054 | 0.049 | -0.000 |
| gemini | Novelty | 90 | 0.168 | 0.109 | -0.000 |
| gemini | Significance | 90 | 0.075 | 0.029 | -0.000 |
| gemini | Testability | 90 | 0.140 | 0.115 | -0.000 |
| gpt | Novelty | 90 | 0.175 | 0.125 | -0.000 |
| gpt | Significance | 90 | 0.029 | 0.044 | -0.000 |
| gpt | Testability | 90 | 0.082 | 0.039 | -0.000 |
