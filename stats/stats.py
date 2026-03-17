"""
stats.py — Statistical reliability analysis of benchmark runs.

Tests whether LLM-as-user produces consistent results across repeated runs.
Uses Krippendorff's alpha and ICC for inter-rater reliability,
plus Kruskal-Wallis to test for significant variance across runs.

Usage:
    python stats.py
    python stats.py --runs-dir runs/
"""

import argparse
import json
import os
import math
from itertools import combinations


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all_runs(runs_dir: str) -> dict:
    """
    Returns a nested dict:
      data[condition][run_id] = {
        "overall": float,
        "by_metric": {mid: float},
        "by_scenario": {sid: float},
        "details": [(scenario_id, metric_id, 0|1)]
      }
    """
    data = {}
    for condition in sorted(os.listdir(runs_dir)):
        cond_path = os.path.join(runs_dir, condition)
        if not os.path.isdir(cond_path):
            continue
        data[condition] = {}
        for run_name in sorted(os.listdir(cond_path)):
            results_path = os.path.join(cond_path, run_name, "results.json")
            if not os.path.exists(results_path):
                continue
            with open(results_path) as f:
                r = json.load(f)
            run_id = run_name  # e.g. "run_1"
            details = [(d["scenario_id"], d["metric_id"], 1 if d["result"] == "pass" else 0)
                       for d in r["details"]]
            data[condition][run_id] = {
                "overall": r["summary"]["pass_rate"],
                "by_metric": {k: v["pass_rate"] for k, v in r["by_metric"].items()},
                "by_scenario": {k: v["pass_rate"] for k, v in r["by_scenario"].items()},
                "details": details,
            }
    return data


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean(xs): return sum(xs) / len(xs)
def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)
def std(xs): return math.sqrt(variance(xs))
def sem(xs): return std(xs) / math.sqrt(len(xs))

def pearson_r(xs, ys):
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx)**2 for x in xs) * sum((y - my)**2 for y in ys))
    return num / den if den else 0.0

def icc_two_way_absolute(ratings: list[list[float]]) -> float:
    """
    ICC(2,1) — two-way random effects, absolute agreement, single measures.
    ratings: list of raters, each a list of scores for N subjects.
    Returns ICC value.
    """
    k = len(ratings)       # number of raters
    n = len(ratings[0])    # number of subjects

    # Grand mean
    grand = mean([x for rater in ratings for x in rater])

    # Between-subjects SS
    subject_means = [mean([ratings[r][i] for r in range(k)]) for i in range(n)]
    ss_between = k * sum((m - grand)**2 for m in subject_means)

    # Within-subjects SS
    ss_within = sum((ratings[r][i] - subject_means[i])**2
                    for r in range(k) for i in range(n))

    # Between-raters SS
    rater_means = [mean(ratings[r]) for r in range(k)]
    ss_raters = n * sum((m - grand)**2 for m in rater_means)

    # Error SS
    ss_error = ss_within - ss_raters

    ms_between = ss_between / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_raters = ss_raters / (k - 1)

    icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + k * (ms_raters - ms_error) / n)
    return round(icc, 4)


def krippendorff_alpha_nominal(ratings: list[list[int]]) -> float:
    """
    Krippendorff's alpha for nominal (binary) data.
    ratings: list of raters, each a list of 0/1 scores.
    """
    k = len(ratings)
    n = len(ratings[0])

    # Observed disagreement
    do = 0.0
    count = 0
    for i in range(n):
        for r1, r2 in combinations(range(k), 2):
            do += (ratings[r1][i] != ratings[r2][i])
            count += 1
    do = do / count if count else 0

    # Expected disagreement (from marginal distribution)
    all_vals = [ratings[r][i] for r in range(k) for i in range(n)]
    p1 = sum(all_vals) / len(all_vals)
    de = 2 * p1 * (1 - p1)

    if de == 0:
        return 1.0
    return round(1 - do / de, 4)


def pairwise_correlations(runs_data: dict) -> list[tuple[str, str, float]]:
    """Pearson r between every pair of runs on their per-item (scenario×metric) vectors."""
    run_ids = sorted(runs_data.keys())
    results = []
    for r1, r2 in combinations(run_ids, 2):
        # Align by (scenario_id, metric_id)
        items1 = {(s, m): v for s, m, v in runs_data[r1]["details"]}
        items2 = {(s, m): v for s, m, v in runs_data[r2]["details"]}
        keys = sorted(set(items1) & set(items2))
        if len(keys) < 2:
            continue
        xs = [items1[k] for k in keys]
        ys = [items2[k] for k in keys]
        r = pearson_r(xs, ys)
        results.append((r1, r2, r))
    return results


def kruskal_wallis_H(groups: list[list[float]]) -> tuple[float, str]:
    """
    Kruskal-Wallis H test (non-parametric ANOVA).
    Returns (H statistic, qualitative interpretation).
    With only 3 runs of 25 items, we report H and note critical value.
    """
    all_vals = sorted([(v, g) for g, grp in enumerate(groups) for v in grp])
    n = len(all_vals)
    ranks = {}
    i = 0
    while i < n:
        j = i
        while j < n and all_vals[j][0] == all_vals[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    n_total = sum(len(g) for g in groups)
    H = 12 / (n_total * (n_total + 1))
    offset = 0
    for grp in groups:
        nj = len(grp)
        rj = sum(ranks[offset + idx] for idx in range(nj))
        H += nj * (rj / nj - (n_total + 1) / 2) ** 2 * (12 / (n_total * (n_total + 1)))
        offset += nj

    # Simplified: recompute properly
    offset = 0
    rank_sums = []
    flat = sorted([(v, g_idx, i_idx)
                   for g_idx, grp in enumerate(groups)
                   for i_idx, v in enumerate(grp)])
    ranked = []
    i = 0
    while i < len(flat):
        j = i
        while j < len(flat) and flat[j][0] == flat[i][0]:
            j += 1
        r = (i + 1 + j) / 2
        for _ in range(i, j):
            ranked.append(r)
        i = j

    flat_groups = [[] for _ in groups]
    for (v, g_idx, _), r in zip(flat, ranked):
        flat_groups[g_idx].append(r)

    N = len(flat)
    H2 = (12 / (N * (N + 1))) * sum(
        len(g) * (mean(g) - (N + 1) / 2) ** 2 for g in flat_groups if g
    )
    # df = k-1; critical value at p=0.05 for df=2 is 5.99
    sig = "p < 0.05 (significant variance)" if H2 > 5.99 else "p > 0.05 (no significant variance)"
    return round(H2, 4), sig


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def analyze_condition(condition: str, runs_data: dict) -> dict:
    run_ids = sorted(runs_data.keys())
    n_runs = len(run_ids)

    overall_rates = [runs_data[r]["overall"] for r in run_ids]

    # All metric IDs
    metric_ids = sorted(runs_data[run_ids[0]]["by_metric"].keys())
    scenario_ids = sorted(runs_data[run_ids[0]]["by_scenario"].keys())

    # Per-item binary vectors for each run (aligned)
    item_keys = sorted({(s, m) for r in run_ids for s, m, _ in runs_data[r]["details"]})
    binary_ratings = []
    for r in run_ids:
        item_map = {(s, m): v for s, m, v in runs_data[r]["details"]}
        binary_ratings.append([item_map.get(k, 0) for k in item_keys])

    # Continuous (pass rate) vectors per metric
    metric_ratings = [[runs_data[r]["by_metric"][m] for r in run_ids] for m in metric_ids]
    scenario_ratings = [[runs_data[r]["by_scenario"][s] for r in run_ids] for s in scenario_ids]

    # ICC on per-item binary scores (treat as continuous for ICC)
    icc = icc_two_way_absolute(binary_ratings) if n_runs >= 2 else None

    # Krippendorff alpha on binary items
    alpha = krippendorff_alpha_nominal(binary_ratings) if n_runs >= 2 else None

    # Pairwise Pearson r
    pair_cors = pairwise_correlations(runs_data)

    # Kruskal-Wallis on overall pass rates (but only 3 data points, so report H only)
    kw_overall = kruskal_wallis_H([[r] for r in overall_rates]) if n_runs >= 3 else None

    # Per-metric stats
    per_metric = {}
    for mid, rates in zip(metric_ids, metric_ratings):
        per_metric[mid] = {
            "runs": rates,
            "mean": round(mean(rates), 4),
            "std": round(std(rates), 4),
            "range": round(max(rates) - min(rates), 4),
        }

    per_scenario = {}
    for sid, rates in zip(scenario_ids, scenario_ratings):
        per_scenario[sid] = {
            "runs": rates,
            "mean": round(mean(rates), 4),
            "std": round(std(rates), 4),
            "range": round(max(rates) - min(rates), 4),
        }

    return {
        "condition": condition,
        "n_runs": n_runs,
        "overall": {
            "runs": overall_rates,
            "mean": round(mean(overall_rates), 4),
            "std": round(std(overall_rates), 4),
            "range": round(max(overall_rates) - min(overall_rates), 4),
            "sem": round(sem(overall_rates), 4),
        },
        "reliability": {
            "icc_2_1": icc,
            "krippendorff_alpha": alpha,
            "pairwise_pearson_r": [(r1, r2, round(r, 4)) for r1, r2, r in pair_cors],
            "mean_pairwise_r": round(mean([r for _, _, r in pair_cors]), 4) if pair_cors else None,
        },
        "kruskal_wallis_overall": kw_overall,
        "per_metric": per_metric,
        "per_scenario": per_scenario,
    }


def print_report(results: list[dict]) -> None:
    icc_interp = lambda v: (
        "poor (<0.50)" if v < 0.50 else
        "moderate (0.50–0.75)" if v < 0.75 else
        "good (0.75–0.90)" if v < 0.90 else
        "excellent (≥0.90)"
    )
    alpha_interp = lambda v: (
        "poor (<0.20)" if v < 0.20 else
        "fair (0.20–0.40)" if v < 0.40 else
        "moderate (0.40–0.60)" if v < 0.60 else
        "substantial (0.60–0.80)" if v < 0.80 else
        "almost perfect (≥0.80)"
    )

    print("=" * 60)
    print("RELIABILITY ANALYSIS: LLM-as-User Benchmark")
    print("=" * 60)
    print()
    print("Reliability benchmarks (standard thresholds):")
    print("  ICC ≥ 0.75 = good | ≥ 0.90 = excellent")
    print("  Krippendorff α ≥ 0.67 = acceptable for research")
    print("  Pearson r ≥ 0.80 = strong correlation between runs")
    print()

    for res in results:
        cond = res["condition"]
        o = res["overall"]
        rel = res["reliability"]

        print(f"{'─' * 60}")
        print(f"Condition: {cond}  ({res['n_runs']} runs)")
        print(f"{'─' * 60}")
        print(f"Overall pass rate:  mean={o['mean']:.1%}  std={o['std']:.1%}  "
              f"range={o['range']:.1%}  SEM={o['sem']:.1%}")
        print(f"Individual runs:    {[f'{r:.1%}' for r in o['runs']]}")
        print()

        icc = rel["icc_2_1"]
        alpha = rel["krippendorff_alpha"]
        mean_r = rel["mean_pairwise_r"]

        print(f"Reliability (on per-item binary pass/fail across {res['n_runs']} runs):")
        print(f"  ICC(2,1):              {icc:.4f}  → {icc_interp(icc)}")
        print(f"  Krippendorff α:        {alpha:.4f}  → {alpha_interp(alpha)}")
        print(f"  Mean pairwise r:       {mean_r:.4f}")
        for r1, r2, r in rel["pairwise_pearson_r"]:
            print(f"    {r1} vs {r2}: r = {r:.4f}")

        if res["kruskal_wallis_overall"]:
            H, sig = res["kruskal_wallis_overall"]
            print(f"  Kruskal-Wallis H:      {H:.4f}  → {sig}")

        print()
        print("  Per-metric consistency (std across runs):")
        for mid, stats in res["per_metric"].items():
            bar = "█" * int(stats["std"] * 50)
            print(f"    {mid}: mean={stats['mean']:.1%}  std={stats['std']:.1%}  "
                  f"range={stats['range']:.1%}  {bar}")

        print()
        print("  Per-scenario consistency (std across runs):")
        for sid, stats in res["per_scenario"].items():
            bar = "█" * int(stats["std"] * 50)
            flag = " ← zero variance" if stats["std"] == 0 else ""
            print(f"    {sid}: mean={stats['mean']:.1%}  std={stats['std']:.1%}  "
                  f"range={stats['range']:.1%}  {bar}{flag}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Statistical reliability analysis of benchmark runs")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    data = load_all_runs(args.runs_dir)
    if not data:
        print(f"No run data found in {args.runs_dir}")
        return

    all_results = []
    for condition, runs_data in sorted(data.items()):
        if len(runs_data) < 2:
            print(f"Skipping {condition}: fewer than 2 runs.")
            continue
        res = analyze_condition(condition, runs_data)
        all_results.append(res)

    print_report(all_results)

    # Save
    out_path = os.path.join(args.runs_dir, "reliability_stats.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full stats saved to: {out_path}")


if __name__ == "__main__":
    main()
