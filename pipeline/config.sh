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
# Jan 2026 test set: 10 tickers × 20 dates (14-col raw LOBSTER preprocessed)
# Must be raw .npy (not encoded tokens) — LOBbench needs real order_ids and prices.
GOOG_DATA="${GOOG_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/GOOG_jan2026}"
AAPL_DATA="${AAPL_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/AAPL_jan2026}"
NVDA_DATA="${NVDA_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/NVDA_jan2026}"
AMZN_DATA="${AMZN_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/AMZN_jan2026}"
META_DATA="${META_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/META_jan2026}"
TSLA_DATA="${TSLA_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/TSLA_jan2026}"
MSFT_DATA="${MSFT_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/MSFT_jan2026}"
AMD_DATA="${AMD_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/AMD_jan2026}"
MU_DATA="${MU_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/MU_jan2026}"
NFLX_DATA="${NFLX_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/NFLX_jan2026}"

# Legacy Jan 2023 paths (kept for reference)
# GOOG_DATA="${GOOG_DATA:-/lus/lfs1aip2/projects/s5e/lob_pipeline/data/GOOG_jan2023_raw}"
# INTC_DATA="${INTC_DATA:-/lus/lfs1aip2/home/s5e/${USER}/LOBS5/data/INTC_jan2023}"

# ============================================================
# Python interpreter (direct path, no conda activation needed)
# ============================================================
# Use shared project-level miniforge (same env training uses)
PYTHON="${PYTHON:-/lus/lfs1aip2/projects/s5e/quant/miniforge3/envs/lobs5/bin/python}"

# ============================================================
# SLURM settings
# ============================================================
PARTITION="workq"
EXCLUDE_NODES="${EXCLUDE_NODES:-nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid010655,nid010052,nid010442,nid010851,nid010499,nid010463}"

# ============================================================
# Notifications (optional)
# ============================================================
# ntfy.sh topics for job completion notifications
# Leave empty to disable notifications
# Set these to receive alerts when jobs complete:
#   export NTFY_TOPIC_INFERENCE="my_inference_topic"
#   export NTFY_TOPIC_BENCHMARKS="my_benchmarks_topic"

NTFY_TOPIC_INFERENCE="${NTFY_TOPIC_INFERENCE:-}"
NTFY_TOPIC_BENCHMARKS="${NTFY_TOPIC_BENCHMARKS:-}"

# Alternative: Read from ~/.ntfy-topic if it exists
if [ -z "$NTFY_TOPIC_INFERENCE" ] && [ -f ~/.ntfy-topic ]; then
    NTFY_TOPIC_INFERENCE=$(cat ~/.ntfy-topic)
    NTFY_TOPIC_BENCHMARKS=$(cat ~/.ntfy-topic)
fi
