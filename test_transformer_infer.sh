#!/bin/bash
#SBATCH --job-name=test-xfmr-infer
#SBATCH --partition=workq
#SBATCH --account=brics.s5e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=test_xfmr_infer_%j.out
#SBATCH --error=test_xfmr_infer_%j.err
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
CKPT_PATH="/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H4-self-attention/checkpoints/j2505280_uc4odetc_2505280"
DATA_DIR="/lus/lfs1aip2/projects/s5e/lob_pipeline/data/GOOG_jan2026"
SAVE_DIR="${REPO_DIR}/LOBS5/inference_results/test_xfmr_${SLURM_JOB_ID}"

echo "=============================================="
echo "Transformer Inference Test (1 node, 1 GPU)"
echo "=============================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Checkpoint:  ${CKPT_PATH}"
echo "Data:        ${DATA_DIR}"
echo "Output:      ${SAVE_DIR}"
echo "=============================================="

# Step 1: Quick import test
cd "${REPO_DIR}/LOBS5"
echo ""
echo "--- Import chain test ---"
$PYTHON -c "
from s5.transformer import TransformerBlock, init_TransformerBlock
print('OK: s5.transformer imports')
from s5.layers import SequenceLayer
print('OK: s5.layers imports')
from s5.seq_model import StackedEncoderModel
print('OK: s5.seq_model imports')
from lob.lob_seq_model import PaddedLobPredModel
print('OK: lob.lob_seq_model imports')
from lob.init_train import init_train_state
print('OK: lob.init_train imports')
print('All imports successful!')
" && echo "Import test PASSED" || { echo "Import test FAILED"; exit 1; }

# Step 2: Run inference (4 sequences, batch_size=4, 1 GPU)
echo ""
echo "--- Running inference ---"
mkdir -p "${SAVE_DIR}"

$PYTHON -u run_inference.py \
    --stock GOOG \
    --ckpt_path "$CKPT_PATH" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --n_cond_msgs 500 \
    --n_gen_msgs 100 \
    --n_sequences 4 \
    --batch_size 4 \
    --checkpoint_step 7001 \
    --rank 0 \
    --world_size 1 \
    2>&1 | tee "${SAVE_DIR}/log.txt"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=============================================="
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: FAILED (exit code $EXIT_CODE)"
    echo "Check log: ${SAVE_DIR}/log.txt"
else
    echo "RESULT: SUCCESS"
    echo "Output: ${SAVE_DIR}"
    for d in data_real data_gen data_cond; do
        count=$(ls "${SAVE_DIR}/${d}/"*message*.csv 2>/dev/null | wc -l)
        echo "  ${d}/: ${count} message files"
    done
fi
echo "=============================================="

# Notify
if command -v ~/bin/slurm-notify &> /dev/null; then
    ~/bin/slurm-notify send "test_xfmr_infer (${SLURM_JOB_ID}) exit=$EXIT_CODE" --topic isambard_training_panfin
fi

exit $EXIT_CODE
