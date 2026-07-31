#!/bin/bash
#SBATCH -J BTL_PyMC
#SBATCH -p cnall
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/btl_%j.out
#SBATCH -e logs/btl_%j.err

cd "${SLURM_SUBMIT_DIR:-$PWD}"
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
  --draws 2000 \
  --tune 4000 \
  --chains 4 \
  --cores 4 \
  --target-accept 0.99 \
  --human-idata-dir human_btl_results \
  --output-dir bayesian_btl_results
