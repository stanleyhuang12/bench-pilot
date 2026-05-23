"""
bootstrap_ci.py — Cluster-bootstrap 95% CIs for per-model Overall scores.

Estimand: metric generalization. The 375 (benchmark, metric) units are
treated as a sample; we resample metrics and recompute each model's
equal-weight Overall (the macro_s_m aggregation that reproduces
finding.md). Polarity is fixed (polarity_map.json); Overall only.

Two resampling variants:
  flat       — 375 metric units, i.i.d. with replacement
  two-stage  — resample benchmarks w/ replacement, then metrics within
               each drawn benchmark w/ replacement (captures
               within-benchmark correlation)

The same resample is applied to all models each iteration, so the
pairwise P(A > B) probabilities are properly paired.

Usage:
    python bootstrap_ci.py
    python bootstrap_ci.py --B 10000 --seed 42 --out bootstrap_ci_results.md
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict

import numpy as np

import reproduce_findings as R


def build_goodness_table():
    """Return (labels, units, bench_of_unit, G) where
    G[m, u] = model m's goodness on metric-unit u (macro_s_m:
    per-metric = mean over scenarios of per-scenario goodness)."""
    pol = R.load_polarity("polarity_map.json")
    benches = R.benchmark_dirs()

    # model -> (bench, mid) -> {"pol", "scn": {sid: [yes, val]}}
    pm = defaultdict(lambda: defaultdict(
        lambda: {"pol": None, "scn": defaultdict(lambda: [0, 0])}))

    for bench, bdir in benches:
        for label, fpath in R.find_model_files(bdir).items():
            data = json.load(open(fpath))
            details = data["details"] if isinstance(data, dict) else data
            for d in details:
                mid = d.get("metric_id")
                p = pol.get((bench, mid))
                if p is None:
                    continue
                sid = d.get("scenario_id", "?")
                cell = pm[label][(bench, mid)]
                cell["pol"] = p
                for r in d.get("results", []):
                    if r == "yes":
                        cell["scn"][sid][0] += 1
                        cell["scn"][sid][1] += 1
                    elif r == "no":
                        cell["scn"][sid][1] += 1

    def metric_goodness(cell):
        gs = []
        for _sid, (y, v) in cell["scn"].items():
            if v:
                rate = y / v
                gs.append(rate if cell["pol"] == "positive" else 1.0 - rate)
        return sum(gs) / len(gs) if gs else None

    labels = sorted(pm)
    # units present for every model
    common = set.intersection(*(set(pm[L]) for L in labels))
    units = sorted(common)  # list of (bench, mid)

    G = np.empty((len(labels), len(units)), dtype=float)
    for i, L in enumerate(labels):
        for j, u in enumerate(units):
            G[i, j] = metric_goodness(pm[L][u])
    return labels, units, G


def flat_indices(rng, n_units, _groups):
    return rng.integers(0, n_units, n_units)


def two_stage_indices(rng, _n_units, groups):
    bkeys = list(groups.keys())
    chosen = rng.integers(0, len(bkeys), len(bkeys))
    out = []
    for c in chosen:
        idx = groups[bkeys[c]]
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def run_variant(name, sampler, labels, groups, G, B, seed):
    rng = np.random.default_rng(seed)
    n_units = G.shape[1]
    boots = np.empty((B, len(labels)), dtype=float)
    for b in range(B):
        idx = sampler(rng, n_units, groups)
        boots[b] = G[:, idx].mean(axis=1)
    point = G.mean(axis=1)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    se = boots.std(axis=0, ddof=1)
    return {"name": name, "point": point, "lo": lo, "hi": hi,
            "se": se, "boots": boots}


def fmt_table(labels, res):
    order = np.argsort(-res["point"])
    lines = [f"### {res['name']}",
             "",
             "| Rank | Model | Overall | 95% CI | SE |",
             "|---|---|---|---|---|"]
    for r, i in enumerate(order, 1):
        lines.append(
            f"| {r} | {labels[i]} | {res['point'][i]:.3f} | "
            f"[{res['lo'][i]:.3f}, {res['hi'][i]:.3f}] | "
            f"{res['se'][i]:.4f} |")
    return "\n".join(lines), order


def adjacent_probs(labels, res, order):
    """P(model ranked r beats model ranked r+1), paired across draws."""
    b = res["boots"]
    rows = ["| Comparison | P(upper > lower) |", "|---|---|"]
    for r in range(len(order) - 1):
        a, c = order[r], order[r + 1]
        p = float((b[:, a] > b[:, c]).mean())
        rows.append(f"| {labels[a]} > {labels[c]} | {p:.3f} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="bootstrap_ci_results.md")
    args = ap.parse_args()

    labels, units, G = build_goodness_table()
    groups = defaultdict(list)
    for j, (bench, _mid) in enumerate(units):
        groups[bench].append(j)
    groups = {k: np.array(v) for k, v in groups.items()}

    print(f"{len(labels)} models, {len(units)} metric units, "
          f"{len(groups)} benchmarks, B={args.B}, seed={args.seed}")

    flat = run_variant("Flat metric bootstrap", flat_indices,
                       labels, groups, G, args.B, args.seed)
    two = run_variant("Two-stage (benchmark → metric) bootstrap",
                      two_stage_indices, labels, groups, G,
                      args.B, args.seed)

    out = ["# Bootstrap 95% CIs — per-model Overall",
           "",
           f"Estimand: metric generalization. B={args.B}, seed={args.seed}, "
           f"percentile method. Polarity fixed (polarity_map.json). "
           f"Aggregation: macro_s_m (matches finding.md ranking).",
           ""]
    for res in (flat, two):
        tbl, order = fmt_table(labels, res)
        out += [tbl, "",
                "Adjacent-rank superiority (paired across resamples):",
                "", adjacent_probs(labels, res, order), ""]
        # consistency check
        d = abs(res["boots"].mean(axis=0) - res["point"]).max()
        out.append(f"_Consistency: max |mean(replicates) − point| "
                   f"= {d:.4f}_")
        out.append("")

    text = "\n".join(out)
    print("\n" + text)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
