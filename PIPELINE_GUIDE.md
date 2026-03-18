# LOBbench Pipeline Guide

Detailed reference for the LOBbench evaluation pipeline: **encoding swap → inference (GPU) → parallel scoring (CPU) → extended metrics → plots**.

## Directory Layout

```
/lus/lfs1aip2/projects/s5e/lob_pipeline/
├── pipeline/
│   ├── run_lobbench_pipeline.sh   # Main entry point (submits 1 SLURM job per stock)
│   ├── _integrated.batch          # 5-phase SLURM batch script
│   ├── config.sh                  # Shared config: Python, data paths, SLURM, excludes
│   └── config.sh.template         # Template for custom configs
├── LOBS5/                         # Local workspace (writable — logs, encoding swap)
│   ├── lob/encoding.py           # Active encoding (copied from canonical repo at job start)
│   └── AlphaTrade/               # gymnax_exchange submodule (orderbook simulator)
├── inference_results/             # Output: data_real/, data_gen/, data_cond/ per run
├── lob_bench/
│   ├── run_bench.py               # Scoring driver (21 metrics × 3 distances, --n_workers 48)
│   ├── run_plotting.py            # Generates plots from score pickles
│   ├── merge_shards.py            # Merges per-metric shard pickles into one file
│   ├── scoring.py                 # Orchestration: binning, grouping, bootstrap CIs
│   ├── metrics.py                 # L1, Wasserstein, KS with JAX vmap bootstrap
│   ├── eval.py                    # Metric functions (spread, OFI, volume, etc.)
│   ├── plotting.py                # Visualization helpers (bar, spider, histogram)
│   └── hf_data_git/               # HuggingFace baseline data (GOOG/INTC Jan 2023 only)
├── data/
│   ├── GOOG_jan2026/              # 20 dates × 2 .npy files (14-col raw LOBSTER)
│   ├── AAPL_jan2026/              # ... 8 tickers total, symlinked from lob_preproc
│   └── GOOG_jan2023_raw/          # Legacy Jan 2023 data
├── bench_data/                    # Symlinked staging for scoring (auto-created)
├── pip_packages/                  # Vendored deps (pandas 3.0, plotly, statsmodels)
├── logs/                          # SLURM stdout/stderr
└── results_<name>/                # Per-run outputs
    ├── scores/*.pkl               # Gzipped pickle: (scores_dict, ci_dict)
    └── plots/*.png                # Distribution comparisons, bar charts, histograms
```

### Checkpoints

```
/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/<wandb-run-name>/
/lus/lfs1aip2/projects/s5e/lob_pipeline/best_checkpoints/<wandb-run-name>/
```

## Quick Start

### 1. Full pipeline (8 tickers, 24tok model)

```bash
cd /lus/lfs1aip2/projects/s5e/lob_pipeline

./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint \
    --stocks "GOOG AAPL NVDA AMZN META TSLA MSFT AMD" \
    --no_hf_compare --n_sequences 1024 \
    --infer_nodes 1 --total_nodes 1 \
    --walltime "04:00:00" --token_mode 24 \
    --name "my_model_24tok"
```

Submits 8 parallel SLURM jobs (1 per stock, 1 node each). Each job:
1. Copies `encoding_24tok.py` → `encoding.py`
2. Runs inference (4 GPUs, ~15 min)
3. Scores 21 metrics with 48 parallel CPU workers (~20 min)
4. Runs extended scoring: conditional, time-lagged, context, divergence (~40 min)
5. Generates plots

### 2. Single ticker, 22tok legacy model

```bash
./pipeline/run_lobbench_pipeline.sh /path/to/ckpt \
    --stocks "GOOG" --token_mode 22 --no_hf_compare \
    --infer_nodes 1 --total_nodes 1 --walltime "04:00:00"
```

### 3. HF-matched mode (GOOG/INTC Jan 2023 only)

```bash
./pipeline/run_lobbench_pipeline.sh /path/to/ckpt
```

Only works for GOOG and INTC with Jan 2023 data paths configured. Generates matched samples for fair comparison against HuggingFace baselines.

### 4. Quick unconditional only (skip extended)

```bash
./pipeline/run_lobbench_pipeline.sh /path/to/ckpt \
    --stocks "GOOG" --no_hf_compare --skip_extended \
    --infer_nodes 1 --total_nodes 1 --walltime "01:00:00"
```

## Token Mode (22tok vs 24tok)

The encoding is swapped at job start based on `--token_mode`:

| Mode | Vocab | Size encoding | Tokens/msg | Source file |
|------|-------|--------------|------------|-------------|
| **24tok** (default) | 2,112 | `size_digit` base-100 (2 tokens) | 24 | `encoding_24tok.py` |
| **22tok** | 12,012 | `size` range(10000) (1 token) | 22 | `encoding_22tok.py` |

**Critical**: Wrong token mode causes `ScopeParamShapeError` (embedding dimension mismatch). Check checkpoint metadata: `cat /path/to/checkpoint/metadata/_ROOT_METADATA | python -m json.tool | grep token_mode`.

## Reading Score Pickles

Score pickles MUST be read with the `lob_bench` env (pandas 3.0.1):

```bash
PYTHONNOUSERSITE=1 /home/s5e/satyamaga.s5e/miniforge3/envs/lob_bench/bin/python3 -c "
import gzip, pickle
with gzip.open('scores_uncond_GOOG_model_integrated_jobid.pkl', 'rb') as f:
    scores, ci = pickle.load(f)
for metric, stats in scores.items():
    if isinstance(stats, dict):
        l1 = stats.get('l1', ('N/A',))[0] if isinstance(stats.get('l1'), tuple) else stats.get('l1', 'N/A')
        print(f'{metric}: L1={l1}')
"
```

pandas 2.x fails with `NotImplementedError: StringDtype` on the `source` column.

## HF-Matched Mode

Only for GOOG/INTC with Jan 2023 data. Creates 3 scoring variants:

| Variant | Description |
|---------|-------------|
| `<model>` | All generated sequences |
| `<model>_matched` | Subset matching HF sample indices (our model) |
| `hf_<model>_matched` | HF baseline on same indices |

For Jan 2026 data or non-GOOG/INTC tickers, use `--no_hf_compare`.

## Environment

- **Python**: `/lus/lfs1aip2/projects/s5e/quant/miniforge3/envs/lobs5/bin/python` (inference + scoring)
- **lob_bench env**: `/home/s5e/satyamaga.s5e/miniforge3/envs/lob_bench/bin/python3` (reading score pickles)
- **PYTHONPATH**: `pip_packages/` added automatically by batch scripts
- **PYTHONNOUSERSITE=1**: Required on compute nodes (prevents stale `~/.local` packages)
- **SLURM**: account `brics.s5e`, partition `workq`

## Known Issues

- **Checkpoint metadata format**: Older checkpoints use `"custom"` key, newer use `"custom_metadata"`. `load_metadata()` handles both.
- **Orbax OCDBT**: Some checkpoints lack aggregate files; TensorStore fallback in `init_train.py`.
- **pandas 3.0**: `groupby().apply()` drops grouping column; fixed in pipeline's `metrics.py`.
- **JAX GPU on login nodes**: Falls back to CPU. Scoring works, just slower bootstrap.
- **Spider plots**: Kaleido needs Chrome; falls back to `.html` on headless nodes.
- **Checkpoint auto-detect**: Only works for checkpoints with numeric step subdirs. Use `--checkpoint_step N` for others.
- **Data format**: Config paths MUST point to 14-col raw LOBSTER `.npy`, NOT encoded tokens. Encoded data causes NaN metrics.
