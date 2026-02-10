#!/bin/bash
# LOBbench Pipeline Configuration
# This file is user-agnostic and will auto-detect paths based on $USER
#
# To override defaults, set environment variables before running:
#   export GOOG_DATA=/path/to/your/goog/data
#   export PYTHON=/path/to/your/python
#   ./pipeline/run_lobbench_pipeline.sh <CKPT_PATH>

# ============================================================
# Data directories (per stock)
# ============================================================
# GOOG: Jan 2023 test set (24tok_preproc format)
# Default location in shared space
GOOG_DATA="${GOOG_DATA:-/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2023/data/test_set/GOOG}"

# INTC: Jan 2023 (proc.npy format, 41-col books padded at runtime)
# Try user's home first, then fallback to shared location
INTC_DATA="${INTC_DATA:-/lus/lfs1aip2/home/s5e/${USER}/LOBS5/data/INTC_jan2023}"
if [ ! -d "$INTC_DATA" ]; then
    INTC_DATA="/lus/lfs1aip2/projects/s5e/lob_pipeline/data/INTC_jan2023"
fi

# ============================================================
# Python interpreter (direct path, no conda activation needed)
# ============================================================
# Auto-detect: Use miniforge3/envs/lobs5 from current user's home
PYTHON="${PYTHON:-/lus/lfs1aip2/home/s5e/${USER}/miniforge3/envs/lobs5/bin/python}"

# Fallback to kangli's environment if current user doesn't have it
if [ ! -f "$PYTHON" ]; then
    PYTHON="/lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/envs/lobs5/bin/python"
fi

# ============================================================
# SLURM settings
# ============================================================
PARTITION="workq"
