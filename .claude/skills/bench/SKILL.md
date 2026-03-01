---
user-invocable: true
description: Benchmark a model checkpoint using the LOBbench pipeline — submits a single integrated SLURM job (inference + scoring + plots)
arguments: "<checkpoint_path_or_name> [options] — checkpoint to benchmark, plus optional flags"
---

# /bench — LOBbench Pipeline

Run the full LOBbench evaluation pipeline for a model checkpoint: **encoding swap → inference (GPU) → parallel scoring (CPU, 48 workers) → extended metrics → plots**, all in a single SLURM job per stock.

Uses the automation pipeline at `/lus/lfs1aip2/projects/s5e/lob_pipeline/`.

**Note**: The pipeline is user-agnostic and works for all s5e group members.
Notifications are optional and user-configurable (see NOTIFICATIONS.md in the pipeline repo).

## Behavior

1. **Resolve the checkpoint**:
   - If given a full path, use it directly
   - If given a name like `logical-serenity-19`, resolve to `/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/<name>*`
   - Also check `best_checkpoints/` in the pipeline repo
   - Verify the checkpoint exists and has step subdirectories

2. **Determine token mode**:
   - Check checkpoint metadata for `token_mode` field (newer checkpoints have it)
   - If not present, infer from model size: d_model=1024 (lobs5v2) → 22tok; d_model=2048 → check vocab
   - Default is 24tok for recent models. When in doubt, ask the user.

3. **Parse options** from the user's request. Map natural language to flags:

   | User says | Flag |
   |-----------|------|
   | "just GOOG" / "GOOG only" | `--stocks "GOOG"` |
   | "all tickers" / "all 8 tickers" | `--stocks "GOOG AAPL NVDA AMZN META TSLA MSFT AMD"` |
   | "no HF comparison" / "custom mode" | `--no_hf_compare` |
   | "512 sequences" / "N samples" | `--no_hf_compare --n_sequences N` |
   | "step 80000" / "use step X" | `--checkpoint_step X` |
   | "call it v5" / "name it v5_remat" | `--name v5_remat` |
   | "22 token" / "legacy encoding" | `--token_mode 22` |
   | "24 token" / "new encoding" | `--token_mode 24` (default) |
   | "batch size 64" | `--batch_size 64` |
   | "300 conditioning messages" | `--n_cond_msgs 300` |
   | "generate 1000 messages" | `--n_gen_msgs 1000` |
   | "skip inference" / "reuse inference from DIR" | `--skip_inference --inference_dir DIR` |
   | "skip extended" / "unconditional only" | `--skip_extended` |
   | "use 4 nodes" / "N nodes for inference" | `--infer_nodes N` |
   | "6 hours" / "walltime 4h" | `--walltime 06:00:00` |
   | "50 total nodes" / "allocate 50 nodes" | `--total_nodes 50` |
   | "1 node per ticker" / "minimal" | `--infer_nodes 1 --total_nodes 1` |

4. **Show the command** before running it. Example:
   ```
   Will run:
     cd /lus/lfs1aip2/projects/s5e/lob_pipeline
     ./pipeline/run_lobbench_pipeline.sh \
         /lus/lfs1aip2/.../checkpoints/j2504227_c4sz78k1_2504227 \
         --checkpoint_step 22872 \
         --name "j2504227_step22872_24tok" \
         --stocks "GOOG AAPL NVDA AMZN META TSLA MSFT AMD" \
         --no_hf_compare --n_sequences 1024 \
         --infer_nodes 1 --total_nodes 1 \
         --walltime "04:00:00" --token_mode 24

   This will submit:
     - 8 integrated jobs (1 per stock), 1 node each, 4h walltime
     - Token mode: 24tok (vocab=2112)
   ```

5. **On confirmation**, run the command and report:
   - Job IDs for each submitted job
   - Expected output location
   - How to monitor: `squeue --me`

6. **Notifications** (optional):
   - Set `NTFY_TOPIC_INFERENCE` and `NTFY_TOPIC_BENCHMARKS` environment variables
   - See `/lus/lfs1aip2/projects/s5e/lob_pipeline/NOTIFICATIONS.md` for setup
   - By default, no notifications are sent

## Two Modes

| Mode | When | What happens |
|------|------|-------------|
| **HF mode** (default) | No `--no_hf_compare` | Matches HuggingFace baseline samples. Only works for GOOG/INTC with Jan 2023 data. |
| **Custom mode** | `--no_hf_compare` | Generates N random samples (default 1024), scores only our model. Works for all 8 tickers. |

**Important**: For Jan 2026 data or tickers other than GOOG/INTC, always use `--no_hf_compare` (no HF baselines exist).

## Token Mode (22tok vs 24tok)

| Mode | Vocab | Size field | Models |
|------|-------|-----------|--------|
| **24tok** (default) | 2,112 | base-100 digits (2 tokens) | j2504227, logical-serenity-19 |
| **22tok** | 12,012 | direct range(10000) (1 token) | twilight-sound-77 (lobs5v2) |

At job start, `_integrated.batch` copies the correct `encoding_{N}tok.py` → `encoding.py`. Wrong token mode causes embedding shape mismatch errors.

## Full CLI Reference

```bash
cd /lus/lfs1aip2/projects/s5e/lob_pipeline

./pipeline/run_lobbench_pipeline.sh <CKPT_PATH> [OPTIONS]

# Required:
#   CKPT_PATH                Checkpoint directory path

# Optional:
#   --name NAME              Run name (default: from checkpoint dirname)
#   --checkpoint_step N      Step to load (default: auto-detect latest)
#   --stocks "GOOG ..."      Stocks to evaluate (default: "GOOG INTC")
#   --batch_size N           Inference batch size (default: 64)
#   --n_cond_msgs N          Conditioning messages (default: 500)
#   --n_gen_msgs N           Generated messages (default: 500)
#   --no_hf_compare          Custom mode: random sampling, no HF comparison
#   --n_sequences N          Custom mode only (default: 1024)
#   --infer_nodes N          Inference nodes (default: 4, each has 4 GPUs)
#   --walltime T             Total walltime (default: 24:00:00)
#   --total_nodes N          Total nodes (default: max(infer_nodes, 20))
#   --skip_inference         Reuse existing inference
#   --skip_extended          Skip extended scoring (conditional, time-lagged, divergence)
#   --inference_dir DIR      Path to existing inference results
#   --token_mode 22|24       Token encoding mode (default: 24)
```

### Examples

```bash
# 8-ticker eval, 24tok model, 1 node per ticker, 4h
./pipeline/run_lobbench_pipeline.sh /path/to/ckpt \
    --stocks "GOOG AAPL NVDA AMZN META TSLA MSFT AMD" \
    --no_hf_compare --n_sequences 1024 \
    --infer_nodes 1 --total_nodes 1 --walltime "04:00:00" \
    --name "my_model_24tok" --token_mode 24

# Single GOOG eval, 22tok legacy model
./pipeline/run_lobbench_pipeline.sh /path/to/ckpt \
    --stocks "GOOG" --no_hf_compare --token_mode 22 \
    --infer_nodes 1 --total_nodes 1 --walltime "04:00:00"

# HF-matched mode (GOOG/INTC Jan 2023 only)
./pipeline/run_lobbench_pipeline.sh /path/to/ckpt

# Quick unconditional only (skip extended scoring)
./pipeline/run_lobbench_pipeline.sh /path/to/ckpt \
    --stocks "GOOG" --no_hf_compare --skip_extended \
    --infer_nodes 1 --total_nodes 1 --walltime "01:00:00"

# Re-score existing inference output
./pipeline/run_lobbench_pipeline.sh /path/to/ckpt \
    --skip_inference --inference_dir /path/to/inference_results/run_GOOG_jobid \
    --stocks "GOOG"
```

### Output Structure

```
LOBS5/inference_results/<name>_<stock>_<jobid>/
  data_real/, data_gen/, data_cond/

lob_pipeline/results_<name>/
  scores/scores_uncond_<STOCK>_<model>_*.pkl    # Unconditional
  scores/scores_cond_<STOCK>_<model>_*.pkl      # Conditional
  scores/scores_time_lagged_<STOCK>_*.pkl       # Time-lagged
  scores/scores_context_<STOCK>_*.pkl           # Context-aware
  scores/scores_div_<STOCK>_*.pkl               # Divergence (sharded)
  plots/*.png                                    # Distribution plots
```

### Timing Estimates (1 node, 1024 sequences, n_workers=48)

| Phase | Time |
|-------|------|
| JIT compilation | ~3-5 min (22tok) / ~10-15 min (360M 24tok) |
| Inference (4 GPUs) | ~10-15 min |
| Unconditional scoring (21 metrics) | ~15-20 min |
| Extended scoring (cond+context+timelag+div) | ~30-40 min |
| Merge + plots | ~5 min |
| **Total per ticker** | **~1h - 1h30m** |

For 8 tickers at 1 node each (8 parallel jobs): **~1h30m wall time, 8-12 node-hours total**.

### Configuration

`pipeline/config.sh` provides shared defaults:
- `PYTHON` — Project-level miniforge: `/lus/lfs1aip2/projects/s5e/quant/miniforge3/envs/lobs5/bin/python`
- `{TICKER}_DATA` — 8 tickers in `data/{TICKER}_jan2026/` (14-col raw .npy, symlinked from lob_preproc)
- `EXCLUDE_NODES` — Known-bad nodes excluded from allocation
- `PARTITION` — `workq`

Override with environment variables:
```bash
export PYTHON=/your/python
export GOOG_DATA=/your/goog/data
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" / exit code 127 | Check `PYTHON` path in config.sh; project miniforge may have moved |
| Embedding shape mismatch | Wrong `--token_mode` — check checkpoint metadata for `token_mode` field |
| `ssm_size_base` AttributeError | Old checkpoint metadata format — `load_metadata()` handles both `custom` and `custom_metadata` keys |
| "No checkpoint steps found" | Pass `--checkpoint_step N` explicitly |
| Inference OOM | Reduce `--batch_size` (default 64, try 32 or 16) |
| 9/20 NaN metrics | Data dir points to encoded tokens, not raw 14-col .npy |
| Job timeout | Increase `--walltime` (1h30m per ticker typical, 4h safe) |
| Permission denied | Ensure you're in `brics.s5e` group |

## Rules

- ALWAYS show the full command before running
- ALWAYS confirm with the user before submitting
- ALWAYS check checkpoint's token mode (22 vs 24) before submitting — wrong mode = wasted job
- For Jan 2026 data or non-GOOG/INTC tickers, always add `--no_hf_compare`
- Report job IDs clearly so the user can monitor with `squeue` or `/jobs`
- Suggest `--infer_nodes 1 --total_nodes 1` for single-ticker eval to minimize node waste
