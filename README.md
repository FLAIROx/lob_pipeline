# LOBbench Pipeline

Automated pipeline for running LOBbench evaluation on LOBS5 checkpoints.

## Directory Structure

```
lob_pipeline/
├── pipeline/              # Pipeline scripts
│   ├── run_lobbench_pipeline.sh  # Main entry point
│   ├── config.sh          # Auto-detected configuration
│   └── config.sh.template # Template for custom configs
├── LOBS5/                 # LOBS5 codebase (inference)
├── lob_bench/             # LOBbench scoring code
├── bench_data/            # Symlink staging for benchmarking
├── logs/                  # SLURM job outputs
└── results_*/             # Benchmark results by run
```

## Quick Start

### 1. Run a benchmark (HF-matched mode, default)
```bash
cd /lus/lfs1aip2/projects/s5e/lob_pipeline
./pipeline/run_lobbench_pipeline.sh \
    /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me
```

This will:
- Generate 3136 GOOG + 1920 INTC sequences matching the HF baseline
- Run LOBbench scoring on all variants (full, matched, HF-matched)
- Save results to `results_<checkpoint_name>/`

### 2. Run custom mode (random samples)
```bash
./pipeline/run_lobbench_pipeline.sh <CKPT_PATH> \
    --no_hf_compare \
    --n_sequences 2048 \
    --stocks "GOOG"
```

## Configuration

The pipeline is **user-agnostic** - it auto-detects paths based on `$USER`.

### Default behavior:
- **Python**: Uses `${USER}`'s miniforge3/envs/lobs5, falls back to kangli's
- **GOOG data**: Shared location at `/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2023/`
- **INTC data**: Tries `${USER}`'s home, falls back to shared pipeline dir

### Override with environment variables:
```bash
export PYTHON=/path/to/your/python
export GOOG_DATA=/path/to/your/goog/data
export INTC_DATA=/path/to/your/intc/data
./pipeline/run_lobbench_pipeline.sh <CKPT_PATH>
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--name NAME` | Run name | Checkpoint dirname |
| `--checkpoint_step N` | Step to load | Auto-detect latest |
| `--stocks "GOOG INTC"` | Stocks to evaluate | Both |
| `--batch_size N` | Inference batch size | 4 |
| `--n_cond_msgs N` | Conditioning messages | 500 |
| `--n_gen_msgs N` | Generated messages | 500 |
| `--no_hf_compare` | Custom mode (random sampling) | HF-matched mode |
| `--n_sequences N` | Custom mode sample count | 1024 |
| `--infer_walltime T` | Inference walltime | 06:00:00 |
| `--bench_walltime T` | Benchmarking walltime | 24:00:00 |
| `--skip_inference` | Reuse existing inference | Run new inference |
| `--inference_dir DIR` | Path to existing results | - |

## Output Structure

```
results_<name>/
├── scores/
│   ├── scores_uncond_GOOG_<model>_*.pkl     # Score pickles
│   └── scores_uncond_INTC_<model>_*.pkl
└── plots/
    ├── summary_stats_comp.png               # Summary comparison
    └── bar_<stock>_<metric>.png            # Per-metric plots
```

## Timing Estimates

| Phase | GOOG (HF, 3136) | INTC (HF, 1920) | Custom (1024) |
|-------|-----------------|-----------------|---------------|
| Inference (GPU) | ~4h | ~2.5h | ~1.5h |
| Scoring (CPU) | ~12-20h | ~8-14h | ~4-8h |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "config.sh not found" | It's auto-generated, should work out of the box |
| "Python not found" | Set `PYTHON` env var or install miniforge3/envs/lobs5 |
| Inference OOM | Reduce `--batch_size` (default 4, minimum 1) |
| Bench job cancelled | Check inference log - bench depends on `afterok` |
| Permission denied | Ensure you're in `brics.s5e` group |

## Group Permissions

This repo is shared among the `brics.s5e` group. All users in s5e can:
- Run the pipeline
- Create result directories
- Access shared data

## Notes

- Plotting may fail for spider plots (requires Chrome/Kaleido), but this is non-fatal
- Notifications are sent to `isambard_inference_panfin` and `isambard_benchmarks_panfin`
- All paths use `/lus/lfs1aip2/projects/s5e/` as the shared space
