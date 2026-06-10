"""
plot_heatmap.py — Render the per-benchmark breakdown as an annotated
PNG heatmap (matplotlib only).

Same finding.md-consistent macro_s_m, (benchmark, metric)-keyed scores as
benchmark_breakdown.py. Models (rows) sorted by Overall; benchmarks
(cols) sorted hardest→easiest; an Overall column is appended.

Usage:
    venv/bin/python3 plot_heatmap.py
    venv/bin/python3 plot_heatmap.py --out heatmap.png --cmap RdYlGn
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bootstrap_ci import build_goodness_table
from benchmark_breakdown import SHORT
import reproduce_findings as R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark_heatmap.png")
    ap.add_argument("--cmap", default="RdYlGn")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    labels, units, G = build_goodness_table()
    present = {b for b, _ in units}
    # original directory ordering (results/ alpha + emotional-dependence last)
    benches = [b for b, _ in R.benchmark_dirs() if b in present]
    cols = {b: [j for j, (bb, _) in enumerate(units) if bb == b]
            for b in benches}

    M = np.array([[G[i, cols[b]].mean() for b in benches]
                  for i in range(len(labels))])
    overall = G.mean(axis=1)

    mo = np.argsort(-overall)                       # best model first
    # benchmarks kept in directory order (no sorting)
    data = np.column_stack([M[mo, :], overall[mo]])
    row_labels = [labels[i] for i in mo]
    col_labels = [SHORT[b] for b in benches] + ["OVERALL"]

    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=(0.62 * n_cols + 3, 0.5 * n_rows + 2))
    im = ax.imshow(data, cmap=args.cmap, vmin=float(data.min()),
                   vmax=float(data.max()), aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    # separate the Overall column visually
    ax.axvline(n_cols - 1.5, color="black", lw=2)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="black" if 0.30 < v < 0.80 else "white")

    ax.set_title("Model performance per-benchmark - All metrics\n"
                 "rows: models by Overall | "
                 "cols: benchmarks in directory order",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("goodness (1 = good)", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}  ({n_rows}×{n_cols})  "
          f"range [{data.min():.3f}, {data.max():.3f}]")


if __name__ == "__main__":
    main()
