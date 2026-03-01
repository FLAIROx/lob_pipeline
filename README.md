# LOBbench Pipeline

Automated pipeline for running LOBbench evaluation on LOBS5 checkpoints. Supports 8 tickers, 22tok/24tok encoding, and multi-node parallel scoring.

## Directory Structure

```
lob_pipeline/
├── pipeline/                      # Pipeline scripts
│   ├── run_lobbench_pipeline.sh   # Main entry point
│   ├── _integrated.batch          # SLURM job: inference → scoring → plots
│   ├── config.sh                  # Shared configuration (Python, data, SLURM)
│   └── config.sh.template         # Template for custom configs
├── LOBS5/                         # S5 model code + inference
│   └── lob/
│       ├── encoding.py            # Active encoding (swapped at job start)
│       ├── encoding_22tok.py      # 22-token encoding (vocab=12012)
│       └── encoding_24tok.py      # 24-token encoding (vocab=2112)
├── lob_bench/                     # LOBbench scoring code
├── data/                          # Real LOBSTER data
│   ├── GOOG_jan2026/              # 20 dates × 2 files (message + orderbook)
│   ├── AAPL_jan2026/              # ... (8 tickers total)
│   └── ...
├── bench_data/                    # Symlink staging (auto-created)
├── logs/                          # SLURM job outputs
└── results_*/                     # Benchmark results by run
```

## Quick Start

### 1. Multi-ticker eval (24tok model, Jan 2026 data)
```bash
cd /lus/lfs1aip2/projects/s5e/lob_pipeline
./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint \
    --stocks "GOOG AAPL NVDA AMZN META TSLA MSFT AMD" \
    --no_hf_compare --n_sequences 1024 \
    --infer_nodes 1 --total_nodes 1 \
    --walltime "04:00:00" --token_mode 24
```

### 2. Single GOOG eval (22tok legacy model)
```bash
./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint \
    --stocks "GOOG" --no_hf_compare --token_mode 22 \
    --infer_nodes 1 --total_nodes 1 --walltime "04:00:00"
```

### 3. HF-matched mode (GOOG/INTC Jan 2023 only)
```bash
./pipeline/run_lobbench_pipeline.sh /path/to/checkpoint
```

## Configuration

`pipeline/config.sh` provides shared defaults:
- **Python**: `/lus/lfs1aip2/projects/s5e/quant/miniforge3/envs/lobs5/bin/python`
- **Data**: 8 tickers in `data/{TICKER}_jan2026/` (14-col raw .npy)
- **SLURM**: partition `workq`, with known-bad nodes excluded

### Override with environment variables:
```bash
export PYTHON=/path/to/your/python
export GOOG_DATA=/path/to/your/goog/data
./pipeline/run_lobbench_pipeline.sh <CKPT_PATH>
```

## Token Mode

| Mode | Vocab | Size field | Flag | Models |
|------|-------|-----------|------|--------|
| **24tok** (default) | 2,112 | base-100 (2 tokens) | `--token_mode 24` | j2504227, logical-serenity-19 |
| **22tok** | 12,012 | direct (1 token) | `--token_mode 22` | twilight-sound-77 (lobs5v2) |

Wrong token mode causes embedding shape mismatches. Check checkpoint metadata for `token_mode` field.

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--name NAME` | Run name | Checkpoint dirname |
| `--checkpoint_step N` | Step to load | Auto-detect latest |
| `--stocks "GOOG ..."` | Stocks to evaluate | `"GOOG INTC"` |
| `--batch_size N` | Inference batch size | 64 |
| `--n_cond_msgs N` | Conditioning messages | 500 |
| `--n_gen_msgs N` | Generated messages | 500 |
| `--no_hf_compare` | Custom mode (random sampling) | HF-matched mode |
| `--n_sequences N` | Custom mode sample count | 1024 |
| `--infer_nodes N` | Inference nodes (4 GPUs each) | 4 |
| `--walltime T` | Total walltime | 24:00:00 |
| `--total_nodes N` | Total nodes to allocate | max(infer_nodes, 20) |
| `--skip_inference` | Reuse existing inference | Run new inference |
| `--skip_extended` | Skip extended scoring | Run all phases |
| `--inference_dir DIR` | Path to existing results | - |
| `--token_mode 22\|24` | Token encoding mode | 24 |

## Output Structure

```
results_<name>/
├── scores/
│   ├── scores_uncond_<STOCK>_<model>_*.pkl     # Unconditional (21 metrics)
│   ├── scores_cond_<STOCK>_<model>_*.pkl       # Conditional
│   ├── scores_time_lagged_<STOCK>_*.pkl        # Time-lagged
│   ├── scores_context_<STOCK>_*.pkl            # Context-aware
│   └── scores_div_<STOCK>_*.pkl                # Divergence
└── plots/*.png                                  # Distribution comparison plots
```

## Timing Estimates (1 node, 1024 sequences)

| Phase | Time |
|-------|------|
| Inference (4 GPUs, incl. JIT) | ~15-20 min |
| Unconditional scoring (21 metrics, 48 workers) | ~15-20 min |
| Extended scoring | ~30-40 min |
| **Total per ticker** | **~1h - 1h30m** |

8 tickers × 1 node each = **~1h30m wall time, 8-12 node-hours total**.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Exit code 127 / "Python not found" | Check `PYTHON` in config.sh |
| Embedding shape mismatch | Wrong `--token_mode` for this checkpoint |
| `ssm_size_base` AttributeError | Old metadata format — already handled |
| 9/20 NaN metrics | Data dir has encoded tokens, not raw 14-col .npy |
| Inference OOM | Reduce `--batch_size` (try 32 or 16) |
| Job timeout | Increase `--walltime` (4h safe for 1 ticker) |
| Permission denied | Ensure you're in `brics.s5e` group |

## Group Permissions

Shared among `brics.s5e` group. All s5e members can run the pipeline, create results, and access shared data.

## Notifications

Optional and user-configurable. See [NOTIFICATIONS.md](NOTIFICATIONS.md).

```bash
export NTFY_TOPIC_INFERENCE="your_topic"
export NTFY_TOPIC_BENCHMARKS="your_topic"
```
