#!/usr/bin/env python3
"""Generate fixed sample indices for LOBbench evaluation.

For each ticker, scans the data directory to count total non-overlapping
sequences, then samples N indices deterministically (seed 42). These
indices ensure every model is evaluated on the exact same sequences.

Usage:
    python generate_sample_indices.py --data_dir data/GOOG_jan2026 --ticker GOOG
    python generate_sample_indices.py --all  # generate for all configured tickers

Output:
    pipeline/sample_indices/{TICKER}_{N}.txt  (one index per line)
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np


def count_sequences(data_dir: str, seq_len: int = 1000) -> tuple[int, list[tuple[str, int, int]]]:
    """Count total non-overlapping sequences across all date files.

    Returns:
        total: total number of sequences
        file_info: list of (filename, n_rows, n_seqs) per date file
    """
    msg_files = sorted(glob(os.path.join(data_dir, '*message*_proc.npy')))
    if not msg_files:
        raise FileNotFoundError(f"No message _proc.npy files in {data_dir}")

    file_info = []
    total = 0
    for f in msg_files:
        # Use mmap to read shape without loading entire file
        arr = np.load(f, mmap_mode='r')
        n_rows = arr.shape[0]
        n_seqs = max(0, n_rows // seq_len)
        file_info.append((os.path.basename(f), n_rows, n_seqs))
        total += n_seqs

    return total, file_info


def generate_indices(total_seqs: int, n_samples: int, seed: int = 42) -> np.ndarray:
    """Sample n_samples indices from [0, total_seqs) with fixed seed."""
    if n_samples > total_seqs:
        print(f"  WARNING: Requested {n_samples} samples but only {total_seqs} "
              f"sequences available. Using all {total_seqs}.")
        n_samples = total_seqs

    rng = np.random.default_rng(seed)
    indices = rng.choice(total_seqs, size=n_samples, replace=False)
    return np.sort(indices)


def main():
    parser = argparse.ArgumentParser(description="Generate fixed sample indices for LOBbench")
    parser.add_argument('--data_dir', type=str, help="Data directory for a single ticker")
    parser.add_argument('--ticker', type=str, help="Ticker symbol")
    parser.add_argument('--all', action='store_true', help="Generate for all configured tickers")
    parser.add_argument('--n_samples', type=int, default=3136, help="Number of samples (default: 3136)")
    parser.add_argument('--seq_len', type=int, default=1000, help="Sequence length (default: 1000)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Output directory (default: pipeline/sample_indices/)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / 'sample_indices'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Read data dirs from config
        repo_dir = script_dir.parent
        tickers_and_dirs = []
        for ticker in ['GOOG', 'AAPL', 'NVDA', 'AMZN', 'META', 'TSLA', 'MSFT', 'AMD', 'MU', 'NFLX']:
            data_dir = repo_dir / 'data' / f'{ticker}_jan2026'
            if data_dir.exists():
                tickers_and_dirs.append((ticker, str(data_dir)))
            else:
                print(f"  SKIP {ticker}: {data_dir} not found")
    elif args.data_dir and args.ticker:
        tickers_and_dirs = [(args.ticker, args.data_dir)]
    else:
        parser.error("Specify --all or both --data_dir and --ticker")
        return

    print(f"Generating {args.n_samples} sample indices (seed={args.seed}, seq_len={args.seq_len})")
    print(f"Output: {out_dir}/")
    print()

    for ticker, data_dir in tickers_and_dirs:
        try:
            total_seqs, file_info = count_sequences(data_dir, args.seq_len)
        except FileNotFoundError as e:
            print(f"  ERROR {ticker}: {e}")
            continue

        indices = generate_indices(total_seqs, args.n_samples, args.seed)
        out_file = out_dir / f'{ticker}_{len(indices)}.txt'
        np.savetxt(out_file, indices, fmt='%d')

        n_dates = len(file_info)
        print(f"  {ticker}: {total_seqs} sequences across {n_dates} dates → "
              f"sampled {len(indices)} → {out_file.name}")

    print(f"\nDone. Index files at: {out_dir}/")


if __name__ == '__main__':
    main()
