#!/bin/bash
#SBATCH --job-name=test-xfmr-diag
#SBATCH --partition=workq
#SBATCH --account=brics.s5e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=test_xfmr_diagnostic_%j.out
#SBATCH --error=test_xfmr_diagnostic_%j.err
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid010655,nid010052,nid010442,nid010851,nid010499,nid010463

set -euo pipefail

export TMPDIR="/tmp"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

REPO_DIR="/lus/lfs1aip2/projects/s5e/lob_pipeline"
PYTHON="/lus/lfs1aip2/projects/s5e/quant/miniforge3/envs/lobs5/bin/python"

echo "=============================================="
echo "Transformer __call_rnn__ Diagnostic"
echo "=============================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "Python:   ${PYTHON}"
echo "=============================================="
echo ""

cd "${REPO_DIR}/LOBS5"
${PYTHON} -u "${REPO_DIR}/test_xfmr_diagnostic.py"
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

# Notify
if command -v ~/bin/slurm-notify &> /dev/null; then
    ~/bin/slurm-notify send "xfmr_diag (${SLURM_JOB_ID}) exit=$EXIT_CODE" --topic isambard_training_panfin
fi

exit $EXIT_CODE
