#!/bin/bash
#SBATCH -J BTL_smoke
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/btl_smoke_%j.out
#SBATCH -e logs/btl_smoke_%j.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs .cache/matplotlib .cache/pytensor

export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
export PYTENSOR_FLAGS="base_compiledir=$PWD/.cache/pytensor"

HPC_CONDA_ROOT="${HPC_CONDA_ROOT:-$HOME/WORK/miniconda3}"
source "$HPC_CONDA_ROOT/bin/activate"
conda activate "$HPC_CONDA_ROOT/envs/LLMpy39"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python run_bayesian_btl_analysis.py \
  --draws 20 \
  --tune 20 \
  --chains 1 \
  --cores 1 \
  --target-accept 0.9 \
  --dimensions Novelty \
  --output-dir bayesian_btl_smoke \
  --skip-existing
