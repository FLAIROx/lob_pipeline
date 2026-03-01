# CLAUDE.md — LOBS5 LOB Pipeline

## What This Is

Automated pipeline for evaluating S5-based generative models of Limit Order Book (LOB) data on HPC (SLURM). Runs inference with trained checkpoints, scores generated sequences against real market data, and produces comparison plots.

## Architecture: Five-Phase Integrated Pipeline

```
encoding swap → inference (GPU) → staging → parallel scoring (CPU) → extended scoring → plots
```

**Entry point**: `./pipeline/run_lobbench_pipeline.sh <CHECKPOINT_PATH> [options]`

One SLURM job is submitted per stock. Each job runs all phases sequentially via `pipeline/_integrated.batch`. Supports 8 tickers: GOOG, AAPL, NVDA, AMZN, META, TSLA, MSFT, AMD.

## Code Structure

| Directory | Purpose |
|-----------|---------|
| `pipeline/` | Orchestration: CLI entry point, SLURM batch scripts, config |
| `LOBS5/` | S5 model code, inference (`run_inference.py`), training |
| `LOBS5/s5/` | S5 SSM core (parallel scan, bilinear discretization) |
| `LOBS5/lob/` | LOB-specific model, tokenization, checkpoint loading |
| `LOBS5/lob/encoding_22tok.py` | 22-token encoding (vocab=12012, single-token size field) |
| `LOBS5/lob/encoding_24tok.py` | 24-token encoding (vocab=2112, base-100 size digits) |
| `lob_bench/` | Scoring engine (`run_bench.py`), plotting (`run_plotting.py`) |
| `lob_bench/hf_data_git/` | HuggingFace baseline data (GOOG/INTC Jan 2023 only) |
| `data/` | Real LOBSTER data: `{TICKER}_jan2026/` (8 tickers × 20 dates) |
| `bench_data/` | Auto-created symlink staging area for scoring |
| `results_*/` | Per-run outputs: `scores/*.pkl` and `plots/*.png` |
| `pip_packages/` | Vendored Python deps (pandas 3.0, plotly, statsmodels, etc.) |
| `logs/` | SLURM job stdout/stderr |

## Key Commands

```bash
# Multi-ticker eval (custom mode, 24tok model)
./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint \
    --stocks "GOOG AAPL NVDA AMZN META TSLA MSFT AMD" \
    --no_hf_compare --n_sequences 1024 \
    --infer_nodes 1 --total_nodes 1 --walltime "04:00:00"

# Single ticker, 22tok model
./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint \
    --stocks "GOOG" --token_mode 22 --no_hf_compare

# HF-matched mode (GOOG/INTC only, Jan 2023 data needed)
./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint

# Skip extended metrics (unconditional only, ~30min per ticker)
./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint --skip_extended

# Skip inference, re-score existing data
./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint \
    --skip_inference --inference_dir /path/to/existing/results

# Monitor running jobs
squeue -u $USER
tail -f logs/integrated_*.out
```

Checkpoints live at: `/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/<wandb-run-name>/`

## Token Mode (22tok vs 24tok)

The pipeline supports two tokenization schemes via `--token_mode`:

| Mode | Vocab size | Size encoding | Tokens/msg | Flag |
|------|-----------|---------------|------------|------|
| **24tok** (default) | 2,112 | base-100 (`size_digit`, 2 tokens) | 24 | `--token_mode 24` |
| **22tok** (legacy) | 12,012 | direct (`size`, 1 token, range 10000) | 22 | `--token_mode 22` |

At job start, `_integrated.batch` copies `encoding_{N}tok.py` → `encoding.py` to match the checkpoint's training vocabulary. Using the wrong token mode causes embedding shape mismatches (e.g. `(12012,2048)` vs `(2112,2048)`).

## Tech Stack

- **Model**: S5 (Structured State Space) via JAX/Flax
- **Inference**: JAX with multi-GPU sharding (srun, 4 GPUs/node)
- **Scoring**: NumPy/SciPy + JAX vmap for GPU-accelerated bootstrap CIs, `--n_workers 48` for CPU parallelism
- **Plotting**: Plotly + Kaleido (PNG export, HTML fallback)
- **Checkpoints**: Orbax with TensorStore OCDBT fallback (see `OCDBT_CHECKPOINT_FIX.md`)
- **Batch system**: SLURM (partition: `workq`, with node exclusion list)
- **Vendored deps**: `pip_packages/` added to PYTHONPATH at runtime

## Configuration System

`pipeline/config.sh` provides shared defaults:
- **Python**: project-level miniforge at `/lus/lfs1aip2/projects/s5e/quant/miniforge3/envs/lobs5/bin/python`
- **Data**: 8 tickers × Jan 2026 in `data/{TICKER}_jan2026/` (14-col raw `.npy`, symlinked from `lob_preproc`)
- **SLURM**: partition `workq`, with `EXCLUDE_NODES` for known-bad nodes
- **Notifications**: optional ntfy.sh topics

**Override with env vars:**
```bash
export PYTHON=/path/to/python
export GOOG_DATA=/path/to/goog
```

## Data Flow

1. **Input**: Real LOBSTER data (14-col raw `.npy` files: `[order_id, event, direction, price, size, ...]`)
2. **Encoding swap**: Copies `encoding_{22|24}tok.py` → `encoding.py` based on `--token_mode`
3. **Preprocessing**: `LOBS5/preproc.py` converts L2 orderbook → 503-dim features (JAX vmapped)
4. **Inference**: `LOBS5/run_inference.py` generates message/orderbook CSVs autoregressively
5. **Staging**: `bench_data/` gets symlinks to inference output + HF baselines (if applicable)
6. **Scoring**: `lob_bench/run_bench.py` computes 21 metrics × 3 distances (L1, Wasserstein, KS) with 48 parallel workers
7. **Output**: Gzipped pickles `(scores_dict, score_dfs_dict)` + PNG plots

### Data Format (CRITICAL)
`config.sh` data paths **MUST** point to 14-col raw LOBSTER `.npy` (int64, real order_ids and prices), **NOT** 22/24-col encoded tokens. Encoded data causes 9/20 NaN metrics because `decode_msg()` replaces order_id and price with -9999.

### Scoring Modes
- **Unconditional** (Phase 3): 21 metrics, sharded across available nodes
- **Extended** (Phase 4): Conditional (spread/volume deciles), context-aware (5-min windows), time-lagged (1-min returns), divergence
- Use `--skip_extended` to skip Phase 4 for faster turnaround

## Pipeline Batch Scripts

| Script | Purpose |
|--------|---------|
| `_integrated.batch` | **Default**: single job doing encoding swap → inference → scoring → extended → plots |
| `_infer.batch` | Legacy: standalone GPU inference |
| `_bench.batch` | Legacy: standalone CPU scoring |
| `_bench_array.batch` | Legacy: sharded scoring array job |

## Checkpoint Compatibility

- **Metadata format**: Newer checkpoints use `custom_metadata` key, older ones use `custom` key. `load_metadata()` in `init_train.py` handles both.
- **Orbax OCDBT**: Some checkpoints lack aggregate files; TensorStore fallback handles this.
- **Auto-detect step**: Only works for checkpoints with numeric step subdirs. Use `--checkpoint_step N` for others.

## HF-Matched Mode

Only available for GOOG and INTC with Jan 2023 data. Generates sequences on the same sample indices as HuggingFace baselines for fair A/B comparison. Creates three scoring variants:
- `{run_name}`: Full dataset scores
- `{run_name}_matched`: Our model on HF indices only
- `hf_{run_name}_matched`: HF model on same indices

For other tickers or Jan 2026 data, use `--no_hf_compare`.

## Known Issues

- **Orbax OCDBT**: Some checkpoints lack aggregate files; TensorStore fallback in `LOBS5/lob/init_train.py`
- **pandas 3.0**: `groupby().apply()` drops grouping column; fixed with `pd.concat()` in `lob_bench/metrics.py`
- **Kaleido/spider plots**: Needs Chrome; falls back to `.html` on headless nodes
- **JAX on login nodes**: No GPU → falls back to CPU (slower but works)
- **PYTHONNOUSERSITE=1**: Required on compute nodes to avoid stale `~/.local` packages shadowing conda

## Existing Documentation

- `README.md` — Quick start, options table, troubleshooting
- `PIPELINE_GUIDE.md` — Detailed workflow walkthrough
- `NOTIFICATIONS.md` — Optional ntfy.sh push notification setup
- `OCDBT_CHECKPOINT_FIX.md` — TensorStore checkpoint loading workaround
