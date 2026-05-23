"""
compare-realism-injection.py — Compare adversarial vs a downsample variant.

Modes (--mode):
  no-adversarial  — conversations/ vs conversations/downsample-no-adversarial/
  no-perfunctory  — conversations/ vs conversations/downsample-no-perfunctory/

Workflow:
  1. Locate both condition directories for the given benchmark.
  2. Filter to only the models listed under models.target in config.json.
  3. Run evaluation for any model dir that lacks results files.
  4. Report combined failure rate (harm_rate) with SEs and a pooled z-test.

Usage:
    python compare-adversarial-design.py --b humanebench
    python compare-adversarial-design.py --b humanebench --mode no-perfunctory
    python compare-adversarial-design.py --b humanebench --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import time

import mcnemar_factorial as _mc
from client import make_client
from config import get_model_name, load_config

# ---------------------------------------------------------------------------
# Import evaluation primitives from 3-evaluation.py via importlib
# (filename starts with a digit so it cannot be directly imported)
# ---------------------------------------------------------------------------
_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3-evaluation.py")
_spec = importlib.util.spec_from_file_location("evaluation", _EVAL_PATH)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

# ---------------------------------------------------------------------------
# Mode configuration — keep suffixes in sync with 2-simulation.py
# ---------------------------------------------------------------------------
# Maps condition name → subdirectory under conversations/.
# "full" means conversations/ itself (no subdir).
_MODE_SUFFIX: dict[str, str] = {
    "full":                          "",
    "no-adversarial":                "downsample-no-adversarial",
    "no-perfunctory":                "downsample-no-perfunctory",
    "no-adversarial-no-perfunctory": "downsample-no-adversarial-no-perfunctory",
}

# All downsample dirs — used to prune os.walk when walking conversations/.
# Must be a superset of _EXCLUDED_CONV_SUBDIRS in 3-evaluation.py.
_ALL_DOWNSAMPLE_SUBDIRS = {"downsample", "downsample-no-adversarial", "downsample-no-perfunctory", "downsample-no-adversarial-no-perfunctory"}


# ---------------------------------------------------------------------------
# Harm-index enrichment
# ---------------------------------------------------------------------------
# Metric dicts in test.json carry no `harm_index`; 3-evaluation.aggregate()
# needs one per metric to compute harm_rate. Mirrors aggregate-results.py's
# _load_harm_index_map: polarity_map.json takes precedence, goal.json fills
# any gaps. Mutates metrics in place AND returns the list for chaining.
def _enrich_harm_index(
    metrics: list[dict],
    bench_dir: str,
    polarity_path: str = "polarity_map.json",
) -> list[dict]:
    benchmark = os.path.basename(os.path.normpath(bench_dir))
    harm_map: dict[str, int | None] = {}

    # 1. polarity_map.json (primary)
    if os.path.exists(polarity_path):
        try:
            with open(polarity_path) as f:
                pol = json.load(f)
            for mid, v in (pol.get(benchmark) or {}).items():
                p = v.get("polarity")
                if p in ("positive", "negative"):
                    harm_map[mid] = 1 if p == "negative" else 0
                elif v.get("harm_index") is not None:
                    harm_map[mid] = v["harm_index"]
        except Exception as e:
            print(f"[warn] failed to read {polarity_path}: {e}")

    # 2. goal.json in bench_dir (fallback, never clobbers a real value)
    goal_path = os.path.join(bench_dir, "goal.json")
    if os.path.exists(goal_path):
        try:
            with open(goal_path) as f:
                goal = json.load(f)
            for m in (goal.get("metric") or goal.get("metrics") or []):
                mid = m.get("id")
                hi  = m.get("harm_index")
                if mid and hi is not None and harm_map.get(mid) is None:
                    harm_map[mid] = hi
        except Exception as e:
            print(f"[warn] failed to read {goal_path}: {e}")

    missing = []
    for m in metrics:
        mid = m.get("id")
        if m.get("harm_index") is None and mid in harm_map:
            m["harm_index"] = harm_map[mid]
        if m.get("harm_index") is None:
            missing.append(mid)
    if missing:
        print(f"[warn] {len(missing)} metric(s) without harm_index in {benchmark}: "
              f"{missing[:5]}{'…' if len(missing) > 5 else ''}")
    return metrics


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _find_model_dirs(
    base: str,
    exclude_subdirs: set[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Recursively find leaf dirs containing scenario_*.json files.
    Returns [(abs_path, rel_model_name), ...].
    Handles flat (gpt-4o/) and nested (anthropic/claude-haiku-4-5/) layouts.
    """
    if not os.path.isdir(base):
        return []
    excluded = exclude_subdirs or set()
    found = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded)
        if any(f.startswith("scenario_") and f.endswith(".json") for f in filenames):
            rel = os.path.relpath(dirpath, base)
            found.append((dirpath, rel))
    return found


def _short_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def _filter_to_config_models(
    model_dirs: list[tuple[str, str]],
    target_models: list[str],
) -> list[tuple[str, str]]:
    """
    Keep only model dirs whose rel name matches a config target model
    (matched by short name or full name).
    """
    allowed_short = {_short_name(m) for m in target_models}
    allowed_full  = set(target_models)
    return [
        (d, rel) for d, rel in model_dirs
        if rel in allowed_full or _short_name(rel) in allowed_short
    ]


def _needs_evaluation(model_dir: str, results_filename: str) -> bool:
    results_path = os.path.join(model_dir, results_filename)
    det_path = _eval._details_path(model_dir, results_filename)
    return not os.path.exists(results_path) and not os.path.exists(det_path)


def _load_summary(model_dir: str, results_filename: str) -> dict | None:
    results_path = os.path.join(model_dir, results_filename)
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        data = json.load(f)
    return data.get("summary") if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _run_evaluation(
    model_dir: str,
    model_rel: str,
    metrics: list[dict],
    client,
    evaluator_model: str,
    results_filename: str,
    max_concurrency: int,
) -> None:
    det_path     = _eval._details_path(model_dir, results_filename)
    results_path = os.path.join(model_dir, results_filename)

    conv_files = sorted(
        fn for fn in os.listdir(model_dir)
        if fn.startswith("scenario_") and fn.endswith(".json")
    )
    if not conv_files:
        print(f"    No conversation files in {model_dir}. Skipping.")
        return

    conversations = []
    for fname in conv_files:
        with open(os.path.join(model_dir, fname)) as f:
            conversations.append(json.load(f))

    pairs = [(conv, metrics) for conv in conversations]
    total_api_calls = sum(len(conv["samples"]) for conv, _ in pairs)
    print(f"    {len(conversations)} conversations · {total_api_calls} API calls · concurrency {max_concurrency}")

    start = time.perf_counter()
    details, tracker, num_failed = asyncio.run(
        _eval.run_evaluations(pairs, client, evaluator_model, max_concurrency)
    )
    elapsed = time.perf_counter() - start

    _eval._save_details(det_path, details)
    results = _eval.aggregate(details, metrics)
    _eval._save_results(results_path, results)

    tracker.write_out_costs(
        step_name="evaluation",
        abs_path_file=model_dir,
        metadata={
            "evaluator_model": evaluator_model,
            "target_model": model_rel,
            "num_conversations": len(conversations),
            "num_metrics": len(metrics),
            "failed_pairs": num_failed,
            "elapsed_seconds": round(elapsed, 2),
        },
    )
    s = results["summary"]
    print(f"    Done in {elapsed:.1f}s  |  cost ${tracker.cost:.4f}  |  {num_failed} failed")
    print(f"    harm_rate: {s['harm_rate']:.1%}")


# ---------------------------------------------------------------------------
# Statistics — pooled two-proportion z-test
# ---------------------------------------------------------------------------

def _se_harm_rate(s: dict) -> float:
    p = s.get("harm_rate", 0.0)
    n = s.get("total_valid", 0)
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def _pooled_z_and_p(
    p_adv: float, n_adv: int,
    p_non: float, n_non: int,
) -> tuple[float, float, float]:
    """
    Pooled two-proportion z-test (H0: p_adv == p_non).
    Returns (z, p_two_tailed, se_pooled).
    """
    if n_adv == 0 or n_non == 0:
        return float("nan"), float("nan"), float("nan")
    p_pool = (p_adv * n_adv + p_non * n_non) / (n_adv + n_non)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_adv + 1 / n_non))
    if se == 0:
        return float("nan"), float("nan"), 0.0
    z = (p_adv - p_non) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p, se


def _sig(p: float) -> str:
    if math.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _cell(v: float | None, se: float | None = None) -> str:
    if v is None:
        return "—"
    s = f"{v:.1%}"
    if se is not None:
        s += f" ±{se:.1%}"
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _cond_path(bench_dir: str, conv_dir_base: str, condition: str) -> str:
    suffix = _MODE_SUFFIX[condition]
    if not suffix:
        return os.path.join(bench_dir, conv_dir_base)
    return os.path.join(bench_dir, conv_dir_base, suffix)


def compare(
    config_path: str,
    results_root: str,
    benchmark: str,
    max_concurrency: int,
    ref: str = "full",
    cmp: str = "no-adversarial",
    plot_visuals: bool = False,
) -> None:
    for cond in (ref, cmp):
        if cond not in _MODE_SUFFIX:
            raise ValueError(f"Unknown condition {cond!r}. Choose from: {list(_MODE_SUFFIX)}")
    if ref == cmp:
        raise ValueError("--ref and --cmp must be different conditions")

    config           = load_config(config_path)
    test_path        = config["paths"]["test_file"]
    conv_dir_base    = config["paths"]["conversations_dir"].rstrip("/")
    results_filename = config["paths"]["results_file"]
    target_models    = get_model_name(config, "target")

    bench_dir = os.path.join(results_root, benchmark)
    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(f"Benchmark not found: {bench_dir}")

    bench_path = os.path.join(bench_dir, test_path)
    if not os.path.exists(bench_path):
        raise FileNotFoundError(f"test.json not found: {bench_path}")

    with open(bench_path) as f:
        test_data = json.load(f)
    metrics: list[dict] = test_data.get("metrics") or test_data.get("metric") or []
    metrics = _enrich_harm_index(metrics, bench_dir)

    client          = make_client(config["models"]["evaluator"])
    evaluator_model = get_model_name(config, "evaluator")

    # ── Locate condition directories ──────────────────────────────────────────
    cond_bases = {
        ref: _cond_path(bench_dir, conv_dir_base, ref),
        cmp: _cond_path(bench_dir, conv_dir_base, cmp),
    }

    for label in list(cond_bases):
        if not os.path.isdir(cond_bases[label]):
            print(f"[skip] {label} directory not found: {cond_bases[label]}")
            del cond_bases[label]

    if not cond_bases:
        print("No condition directories found. Run 2-simulation.py first.")
        return

    # ── Discover model dirs, filtered to config target models ─────────────────
    cond_model_dirs: dict[str, list[tuple[str, str]]] = {}

    for label, base in cond_bases.items():
        exclude = _ALL_DOWNSAMPLE_SUBDIRS if label == "full" else None
        all_dirs = _find_model_dirs(base, exclude_subdirs=exclude)
        model_dirs = _filter_to_config_models(all_dirs, target_models)

        cond_model_dirs[label] = model_dirs
        print(f"\n{'='*62}")
        print(f"Condition : {label}")
        print(f"Directory : {base}")
        print(f"Models    : {[rel for _, rel in model_dirs] or '(none matched config)'}")
        print(f"{'='*62}")

        for model_dir, model_rel in model_dirs:
            if _needs_evaluation(model_dir, results_filename):
                print(f"\n  [{label} / {model_rel}] No results — evaluating...")
                _run_evaluation(
                    model_dir, model_rel, metrics,
                    client, evaluator_model, results_filename, max_concurrency,
                )
            else:
                print(f"  [{label} / {model_rel}] Results exist — skipping.")

    # ── Build summaries (only models present in both conditions) ──────────────
    all_models = sorted(
        rel for rel in {rel for dirs in cond_model_dirs.values() for _, rel in dirs}
        if all(
            any(r == rel for _, r in cond_model_dirs.get(label, []))
            for label in cond_bases
        )
    )

    summaries: dict[str, dict[str, dict | None]] = {}
    for model_rel in all_models:
        summaries[model_rel] = {}
        for label, model_dirs in cond_model_dirs.items():
            match = next((d for d, r in model_dirs if r == model_rel), None)
            summaries[model_rel][label] = _load_summary(match, results_filename) if match else None

    # ── Print comparison table ────────────────────────────────────────────────
    W_MODEL = 42
    W_CELL  = 22

    print(f"\n{'═'*96}")
    print(f"  COMBINED FAILURE RATE  (harm_rate)  —  {ref}  vs  {cmp}")
    print(f"{'═'*96}")
    header = (
        f"{'Model':<{W_MODEL}}"
        f"{ref:>{W_CELL}}"
        f"{cmp:>{W_CELL}}"
        f"{'Δ (cmp−ref)':>{W_CELL}}"
        f"  {'Z':>7}"
        f"  {'p':>8}"
    )
    print(header)
    print("─" * len(header))

    for model_rel in all_models:
        s_ref = summaries[model_rel].get(ref)
        s_cmp = summaries[model_rel].get(cmp)

        p_ref  = s_ref["harm_rate"]   if s_ref else None
        p_cmp  = s_cmp["harm_rate"]   if s_cmp else None
        se_ref = _se_harm_rate(s_ref) if s_ref else None
        se_cmp = _se_harm_rate(s_cmp) if s_cmp else None
        n_ref  = s_ref["total_valid"] if s_ref else 0
        n_cmp  = s_cmp["total_valid"] if s_cmp else 0

        row = f"{model_rel:<{W_MODEL}}"
        row += f"{_cell(p_ref, se_ref):>{W_CELL}}"
        row += f"{_cell(p_cmp, se_cmp):>{W_CELL}}"

        if p_ref is not None and p_cmp is not None:
            diff = p_cmp - p_ref
            z, p, se_pool = _pooled_z_and_p(p_cmp, n_cmp, p_ref, n_ref)
            stars = _sig(p)
            row += f"{_cell(diff, se_pool):>{W_CELL}}  {z:>7.2f}  {p:>7.4f}{stars}"
        else:
            row += f"{'—':>{W_CELL}}  {'—':>7}  {'—':>7}"

        print(row)

    print(f"\n{'─'*96}")
    print(f"Δ = {cmp} − {ref}  (positive = {cmp} has higher failure rate)")
    print("SE shown per condition: √(p·(1−p)/n)   |   Z-test: pooled two-proportion")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  (two-tailed)")
    print()

    if plot_visuals:
        _plot_comparison(all_models, summaries, ref, cmp, benchmark)


def _plot_comparison(
    all_models: list[str],
    summaries: dict[str, dict[str, dict | None]],
    ref: str,
    cmp: str,
    benchmark: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    COLOR_REF = "#90caf9"   # light blue — reference condition
    COLOR_CMP = "#e57373"   # red        — comparison condition

    short_names = [_short_name(m) for m in all_models]
    n = len(all_models)

    p_ref_vals, se_ref_vals = [], []
    p_cmp_vals, se_cmp_vals = [], []
    sig_labels = []

    for model_rel in all_models:
        s_ref = summaries[model_rel].get(ref)
        s_cmp = summaries[model_rel].get(cmp)

        p_ref  = s_ref["harm_rate"]   if s_ref else 0.0
        p_cmp  = s_cmp["harm_rate"]   if s_cmp else 0.0
        se_ref = _se_harm_rate(s_ref) if s_ref else 0.0
        se_cmp = _se_harm_rate(s_cmp) if s_cmp else 0.0
        n_ref  = s_ref["total_valid"] if s_ref else 0
        n_cmp  = s_cmp["total_valid"] if s_cmp else 0

        p_ref_vals.append(p_ref * 100)
        p_cmp_vals.append(p_cmp * 100)
        se_ref_vals.append(se_ref * 100)
        se_cmp_vals.append(se_cmp * 100)

        if s_ref and s_cmp:
            _, p_val, _ = _pooled_z_and_p(p_cmp, n_cmp, p_ref, n_ref)
            sig_labels.append(_sig(p_val))
        else:
            sig_labels.append("")

    x = np.arange(n)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 1.6), 6))

    ax.bar(x - bar_w / 2, p_ref_vals, bar_w,
           color=COLOR_REF, label=ref.replace("-", " ").title(),
           yerr=se_ref_vals, capsize=4,
           error_kw={"linewidth": 1.2, "ecolor": "#555"})

    ax.bar(x + bar_w / 2, p_cmp_vals, bar_w,
           color=COLOR_CMP, label=cmp.replace("-", " ").title(),
           yerr=se_cmp_vals, capsize=4,
           error_kw={"linewidth": 1.2, "ecolor": "#555"})

    for i, stars in enumerate(sig_labels):
        if not stars:
            continue
        top = max(p_ref_vals[i] + se_ref_vals[i], p_cmp_vals[i] + se_cmp_vals[i])
        ax.text(x[i], top + 0.8, stars, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222")

    all_tops = (
        [v + se for v, se in zip(p_ref_vals, se_ref_vals)]
        + [v + se for v, se in zip(p_cmp_vals, se_cmp_vals)]
    )
    y_max = max(all_tops) if all_tops else 10.0
    ax.set_ylim(0, y_max * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Combined Failure Rate (%)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.7)

    fig.suptitle(
        f"Comparison: {ref}  vs  {cmp}",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(f"Benchmark: {benchmark}", fontsize=9, color="#555", pad=10)

    plt.tight_layout()
    out = f"comparison_{ref}_vs_{cmp}_{benchmark}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# All-conditions comparison
# ---------------------------------------------------------------------------

# Ordered from cleanest baseline to most stimulated — determines bar order & color.
_ALL_CONDITIONS_ORDER = [
    "no-adversarial-no-perfunctory",
    "no-adversarial",
    "no-perfunctory",
    "full",
]

_CONDITION_COLORS = {
    "no-adversarial-no-perfunctory": "#90caf9",  # light blue
    "no-adversarial":                "#81c784",  # green
    "no-perfunctory":                "#ffb74d",  # orange
    "full":                          "#e57373",  # red
}

_CONDITION_LABELS = {
    "no-adversarial-no-perfunctory": "Neither",
    "no-adversarial":                "Perfunctory only",
    "no-perfunctory":                "Adversarial only",
    "full":                          "Both",
}

# Key pairwise tests to annotate: (ref, cmp, label)
_KEY_CONTRASTS = [
    ("no-adversarial-no-perfunctory", "no-perfunctory", "adv effect\n(perf=OFF)"),
    ("no-adversarial",                "full",            "adv effect\n(perf=ON)"),
    ("no-adversarial-no-perfunctory", "no-adversarial",  "perf effect\n(adv=OFF)"),
    ("no-perfunctory",                "full",            "perf effect\n(adv=ON)"),
]


def compare_all(
    config_path: str,
    results_root: str,
    benchmark: str,
    max_concurrency: int,
    plot_visuals: bool = False,
) -> None:
    config           = load_config(config_path)
    test_path        = config["paths"]["test_file"]
    conv_dir_base    = config["paths"]["conversations_dir"].rstrip("/")
    results_filename = config["paths"]["results_file"]
    target_models    = get_model_name(config, "target")

    bench_dir = os.path.join(results_root, benchmark)
    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(f"Benchmark not found: {bench_dir}")

    bench_path = os.path.join(bench_dir, test_path)
    with open(bench_path) as f:
        test_data = json.load(f)
    metrics: list[dict] = test_data.get("metrics") or test_data.get("metric") or []
    metrics = _enrich_harm_index(metrics, bench_dir)

    client          = make_client(config["models"]["evaluator"])
    evaluator_model = get_model_name(config, "evaluator")

    # ── Load each condition ───────────────────────────────────────────────────
    cond_model_dirs: dict[str, list[tuple[str, str]]] = {}
    available: list[str] = []

    for cond in _ALL_CONDITIONS_ORDER:
        base = _cond_path(bench_dir, conv_dir_base, cond)
        if not os.path.isdir(base):
            print(f"[skip] {cond} not found: {base}")
            continue
        available.append(cond)
        exclude = _ALL_DOWNSAMPLE_SUBDIRS if cond == "full" else None
        all_dirs = _find_model_dirs(base, exclude_subdirs=exclude)
        model_dirs = _filter_to_config_models(all_dirs, target_models)
        cond_model_dirs[cond] = model_dirs

        print(f"\n{'='*62}")
        print(f"Condition : {cond}")
        print(f"Models    : {[rel for _, rel in model_dirs] or '(none matched config)'}")
        print(f"{'='*62}")

        for model_dir, model_rel in model_dirs:
            if _needs_evaluation(model_dir, results_filename):
                print(f"\n  [{cond} / {model_rel}] No results — evaluating...")
                _run_evaluation(
                    model_dir, model_rel, metrics,
                    client, evaluator_model, results_filename, max_concurrency,
                )
            else:
                print(f"  [{cond} / {model_rel}] Results exist — skipping.")

    if not available:
        print("No condition directories found.")
        return

    # Only keep models present in ALL available conditions
    all_models = sorted(
        rel for rel in {rel for dirs in cond_model_dirs.values() for _, rel in dirs}
        if all(any(r == rel for _, r in cond_model_dirs.get(c, [])) for c in available)
    )

    summaries: dict[str, dict[str, dict | None]] = {}
    for model_rel in all_models:
        summaries[model_rel] = {}
        for cond, model_dirs in cond_model_dirs.items():
            match = next((d for d, r in model_dirs if r == model_rel), None)
            summaries[model_rel][cond] = _load_summary(match, results_filename) if match else None

    # ── Print table ───────────────────────────────────────────────────────────
    W_MODEL = 36
    W_CELL  = 18
    cols = available

    print(f"\n{'═'*110}")
    print(f"  COMBINED FAILURE RATE (harm_rate) — all conditions — {benchmark}")
    print(f"{'═'*110}")
    header = f"{'Model':<{W_MODEL}}" + "".join(f"{_CONDITION_LABELS[c]:>{W_CELL}}" for c in cols)
    print(header)
    print("─" * len(header))

    for model_rel in all_models:
        row = f"{_short_name(model_rel):<{W_MODEL}}"
        for cond in cols:
            s = summaries[model_rel].get(cond)
            p  = s["harm_rate"]   if s else None
            se = _se_harm_rate(s) if s else None
            row += f"{_cell(p, se):>{W_CELL}}"
        print(row)

    # ── Print key contrasts ───────────────────────────────────────────────────
    print(f"\n{'─'*110}")
    print("  KEY CONTRASTS")
    print(f"{'─'*110}")
    avail_contrasts = [(r, c, lbl) for r, c, lbl in _KEY_CONTRASTS if r in available and c in available]
    c_header = f"{'Model':<{W_MODEL}}" + "".join(f"{lbl.replace(chr(10),' '):>{W_CELL}}" for _, _, lbl in avail_contrasts)
    print(c_header)
    print("─" * len(c_header))

    for model_rel in all_models:
        row = f"{_short_name(model_rel):<{W_MODEL}}"
        for ref, cmp, _ in avail_contrasts:
            s_ref = summaries[model_rel].get(ref)
            s_cmp = summaries[model_rel].get(cmp)
            if s_ref and s_cmp:
                diff = s_cmp["harm_rate"] - s_ref["harm_rate"]
                z, p, se_pool = _pooled_z_and_p(
                    s_cmp["harm_rate"], s_cmp["total_valid"],
                    s_ref["harm_rate"], s_ref["total_valid"],
                )
                stars = _sig(p)
                row += f"{_cell(diff, se_pool) + stars:>{W_CELL}}"
            else:
                row += f"{'—':>{W_CELL}}"
        print(row)

    print(f"\n{'─'*110}")
    print("Δ = cmp − ref  (positive = cmp has higher failure rate)")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  (two-tailed, pooled z-test)")
    print()

    # Pooled paired McNemar contrasts (intersection of scenario IDs across cells).
    # Reuses mcnemar_factorial primitives; failure here shouldn't kill the report.
    try:
        mc_rows, mc_n, mc_outcomes = _compute_pooled_mcnemar(
            bench_dir, cond_model_dirs, all_models, available, results_filename,
        )
    except Exception as e:
        print(f"[warn] McNemar computation failed: {e}")
        mc_rows, mc_n, mc_outcomes = None, 0, {}

    # Always emit a plaintext report alongside the run — cheap, scriptable artifact.
    out_txt = os.path.join(bench_dir, "realism_injection.txt")
    _write_realism_injection_txt(
        out_txt, all_models, summaries, available, benchmark,
        mcnemar_rows=mc_rows, mcnemar_n_scenarios=mc_n,
        mcnemar_outcomes=mc_outcomes,
    )
    print(f"Wrote {out_txt}")

    if plot_visuals:
        _plot_all_conditions(all_models, summaries, available, benchmark)


# ---------------------------------------------------------------------------
# Plaintext report
# ---------------------------------------------------------------------------

def _compute_pooled_mcnemar(
    bench_dir: str,
    cond_model_dirs: dict[str, list[tuple[str, str]]],
    all_models: list[str],
    available: list[str],
    results_filename: str,
) -> tuple[list[tuple[str, str, str, dict]], int, dict[str, dict[str, dict]]] | tuple[None, 0, dict]:
    """
    For each key contrast that has both conditions present, run a paired McNemar
    test pooled across `all_models` on the scenario IDs common to all available
    cells. Returns (rows, n_scenarios, outcomes) where:
        rows     = [(ref, cmp, label, result_dict)]      pooled-across-models McNemar
        outcomes = {cond: {model_rel: {(sid,mid,k): 0|1}}}  for per-model derivations
    Returns (None, 0, {}) if fewer than two conditions are available.
    """
    avail = [(r, c, lbl) for r, c, lbl in _KEY_CONTRASTS if r in available and c in available]
    if not avail or len(available) < 2:
        return None, 0, {}

    harm_map = _mc._harm_index_map(bench_dir)

    # Intersection of scenario_ids across every (condition × model) cell that
    # appears in our analysis set — same convention as mcnemar_factorial.run().
    per_cell_sids: list[set[str]] = []
    for cond in available:
        cell: set[str] = set()
        for d, rel in cond_model_dirs.get(cond, []):
            if rel not in all_models:
                continue
            for fn in os.listdir(d):
                if fn.startswith("scenario_") and fn.endswith(".json"):
                    try:
                        jd = json.load(open(os.path.join(d, fn)))
                        sid = jd.get("scenario_id") or jd.get("id")
                        if sid:
                            cell.add(sid)
                    except Exception:
                        pass
        per_cell_sids.append(cell)
    keep_sids = set.intersection(*per_cell_sids) if per_cell_sids else set()

    # outcomes[cond][model_rel] = {(sid, mid, k): harm_0_or_1}
    outcomes: dict[str, dict[str, dict]] = {}
    for cond in available:
        outcomes[cond] = {}
        for d, rel in cond_model_dirs.get(cond, []):
            if rel not in all_models:
                continue
            details = _mc._load_details(d, results_filename)
            outcomes[cond][rel] = _mc._harm_outcomes(details, harm_map, keep_sids)

    rows: list[tuple[str, str, str, dict]] = []
    for ref, cmp_c, lbl in avail:
        # Prefix by model so identical (sid,mid,k) across models stay distinct.
        out_ref = {(m, *k): v for m in all_models for k, v in outcomes[ref].get(m, {}).items()}
        out_cmp = {(m, *k): v for m in all_models for k, v in outcomes[cmp_c].get(m, {}).items()}
        rows.append((ref, cmp_c, lbl, _mc._contrast(out_ref, out_cmp)))
    return rows, len(keep_sids), outcomes


def _pooled_counts(summaries: dict[str, dict[str, dict | None]],
                   models: list[str], cond: str) -> tuple[int, int]:
    """Sum (harm, valid) across `models` for condition `cond`. Skips missing cells."""
    harm = valid = 0
    for m in models:
        s = summaries[m].get(cond)
        if not s:
            continue
        valid += s.get("total_valid", 0)
        harm  += s.get("total_harm", 0)
    return harm, valid


def _write_mcnemar_digest(
    w,
    all_models: list[str],
    mcnemar_rows: list[tuple[str, str, str, dict]],
    mcnemar_outcomes: dict[str, dict[str, dict]],
) -> None:
    """
    Plain-language summary of the McNemar tables. Written so a reader can skim
    one block and walk away with the headline + interaction + per-model picture
    without having to read the raw n01/n10 tables above.
    """
    # Index pooled rows by label for easy lookup.
    by_lbl = {lbl.replace("\n", " "): (ref, cmp_c, r) for ref, cmp_c, lbl, r in mcnemar_rows}

    def _fmt_p(p: float) -> str:
        if math.isnan(p):
            return "n/a"
        return "<.001" if p < 0.001 else f"={p:.3f}"

    w("=" * 96)
    w("  Reader's digest — McNemar paired analysis")
    w("=" * 96)
    w("What the test asks")
    w("------------------")
    w("For each of two simulation knobs (adversarial framing, perfunctory persona), does")
    w("turning it ON change the per-item harm verdict on the SAME scenario/metric/sample")
    w("more often in the 'harm' direction than in the 'not-harm' direction? Each scenario-")
    w("metric-sample is judged under both conditions; we count flips.")
    w("")

    # Headline: biggest paired effect.
    if by_lbl:
        biggest_lbl, (_, _, biggest) = max(
            by_lbl.items(),
            key=lambda kv: (-(kv[1][2]["p"] if not math.isnan(kv[1][2]["p"]) else 1.0),
                            abs(kv[1][2]["delta"])),
        )
        w("Headline")
        w("--------")
        w(f"The largest paired effect is '{biggest_lbl}': Δ = {biggest['delta']:+.1%} on "
          f"{biggest['n_pairs']:,} matched judgments (p {_fmt_p(biggest['p'])}).")
        w(f"  - {biggest['n01']:,} items flipped from no-harm → harm when the knob was added")
        w(f"  - {biggest['n10']:,} items flipped the other way")
        w(f"  - net: ~{biggest['n01'] - biggest['n10']:,} more harm verdicts induced by the change")
        w("")

    # Interaction story — compare effect when other knob OFF vs ON.
    pairs = [
        ("Adversarial framing",
         "adv effect (perf=OFF)", "adv effect (perf=ON)"),
        ("Perfunctory persona",
         "perf effect (adv=OFF)", "perf effect (adv=ON)"),
    ]
    w("Interaction (the punchline)")
    w("---------------------------")
    w("Each knob is essentially inert on its own, but layering one on top of the other")
    w("triggers a large jump in harm verdicts. The two stimulants amplify each other:")
    w("")
    for knob, off_lbl, on_lbl in pairs:
        if off_lbl in by_lbl and on_lbl in by_lbl:
            _, _, off_r = by_lbl[off_lbl]
            _, _, on_r  = by_lbl[on_lbl]
            w(f"  {knob}:")
            w(f"    alone (other knob OFF):  Δ = {off_r['delta']:+.1%}  p {_fmt_p(off_r['p'])}")
            w(f"    on top of other knob:    Δ = {on_r['delta']:+.1%}  p {_fmt_p(on_r['p'])}")
    w("")

    # Per-model picture — separate models that drive the effect from flat ones.
    w("Per-model picture")
    w("-----------------")
    w("We split models by whether either of the 'on top of other knob' contrasts reaches")
    w("p<.001 paired. That cleanly separates models the design moves from models it doesn't.")
    w("")
    on_top_lbls = ["adv effect (perf=ON)", "perf effect (adv=ON)"]
    available_on_top = [lbl for lbl in on_top_lbls if lbl in by_lbl]

    movers: list[tuple[str, list[tuple[str, dict]]]] = []
    flat:   list[tuple[str, list[tuple[str, dict]]]] = []
    for m in all_models:
        per_contrast: list[tuple[str, dict]] = []
        any_sig = False
        for lbl in available_on_top:
            ref, cmp_c, _ = by_lbl[lbl]
            r = _mc._contrast(
                mcnemar_outcomes.get(ref, {}).get(m, {}),
                mcnemar_outcomes.get(cmp_c, {}).get(m, {}),
            )
            per_contrast.append((lbl, r))
            if not math.isnan(r["p"]) and r["p"] < 0.001:
                any_sig = True
        (movers if any_sig else flat).append((m, per_contrast))

    if movers:
        w(f"Driven by the design ({len(movers)} of {len(all_models)}):")
        for m, rows in movers:
            cells = "  |  ".join(
                f"{lbl.split('(')[0].strip()}: Δ {r['delta']:+.1%} (p {_fmt_p(r['p'])})"
                for lbl, r in rows
            )
            w(f"  - {_short_name(m):<30} {cells}")
        w("")
    if flat:
        w(f"Essentially unmoved ({len(flat)} of {len(all_models)}):")
        for m, rows in flat:
            cells = "  |  ".join(
                f"{lbl.split('(')[0].strip()}: Δ {r['delta']:+.1%} (p {_fmt_p(r['p'])})"
                for lbl, r in rows
            )
            w(f"  - {_short_name(m):<30} {cells}")
        w("")

    w("What to take from this")
    w("----------------------")
    w("- The 'full' production setup (both stimulants ON) is materially more aggressive")
    w("  than any single-knob variant. Reported harm rates partly reflect the layered")
    w("  stimulation, not just baseline model behaviour.")
    w("- Frontier models (the 'unmoved' group) are robust to the layered design; their")
    w("  harm rates are about the same whatever you do.")
    w("- For ablation studies that want to isolate one knob's contribution, the right")
    w("  comparison is between two cells that differ on exactly one knob (the paired")
    w("  contrasts above), not 'full' vs the smallest cell.")
    w("")


def _write_realism_injection_txt(
    out_path: str,
    all_models: list[str],
    summaries: dict[str, dict[str, dict | None]],
    available: list[str],
    benchmark: str,
    mcnemar_rows: list[tuple[str, str, str, dict]] | None = None,
    mcnemar_n_scenarios: int = 0,
    mcnemar_outcomes: dict[str, dict[str, dict]] | None = None,
) -> None:
    """
    Write a self-contained plaintext summary of the 4-condition realism-injection
    analysis: per-model harm rates, per-model pairwise contrasts on the 4 key
    effects, and a models-pooled contrast that answers "is the effect detectable
    in aggregate."
    """
    from datetime import datetime, timezone

    # Order condition columns by the canonical baseline→stimulated order
    cond_cols = [c for c in _ALL_CONDITIONS_ORDER if c in available]
    contrasts = [(r, c, lbl) for r, c, lbl in _KEY_CONTRASTS if r in available and c in available]

    W_MODEL = 48
    W_CELL  = 18

    lines: list[str] = []
    w = lines.append

    w("=" * 96)
    w(f"  Realism-injection analysis — {benchmark}")
    w("=" * 96)
    w(f"Generated:        {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    w(f"Benchmark:        {benchmark}")
    w(f"Conditions:       {', '.join(cond_cols)}")
    w(f"Models compared:  {len(all_models)} (must appear in every condition)")
    w("")
    w("Design")
    w("------")
    w("2×2 factorial over two simulation knobs:")
    w("  - adversarial : user-prompt is constructed with awareness of the metrics")
    w("  - perfunctory : user adopts a terse, low-effort persona")
    w("")
    w("Conditions:")
    for c in cond_cols:
        w(f"  {_CONDITION_LABELS[c]:<18} ({c})")
    w("")
    w("Outcome: harm_rate = total_harm / total_valid across all judged metric calls.")
    w("Per-cell SE = sqrt(p·(1−p)/n). Contrast SE pooled under H0 (two-proportion z).")
    w("")

    # ── 1. Per-model rates across conditions ─────────────────────────────────
    w("=" * 96)
    w("  Per-model harm rate by condition")
    w("=" * 96)
    header = f"{'Model':<{W_MODEL}}" + "".join(f"{_CONDITION_LABELS[c]:>{W_CELL}}" for c in cond_cols)
    w(header)
    w("-" * len(header))
    for m in all_models:
        row = f"{m:<{W_MODEL}}"
        for c in cond_cols:
            s = summaries[m].get(c)
            row += f"{_cell(s['harm_rate'] if s else None, _se_harm_rate(s) if s else None):>{W_CELL}}"
        w(row)

    # n per cell, for transparency
    w("")
    w("Sample sizes (total_valid per cell)")
    w("-" * len(header))
    for m in all_models:
        row = f"{m:<{W_MODEL}}"
        for c in cond_cols:
            s = summaries[m].get(c)
            row += f"{(s['total_valid'] if s else 0):>{W_CELL}}"
        w(row)
    w("")

    # ── 2. Per-model pairwise contrasts (the 4 key effects) ──────────────────
    w("=" * 96)
    w("  Per-model pairwise contrasts (pooled two-proportion z-test)")
    w("=" * 96)
    W_CONTRAST = 22
    c_header = f"{'Model':<{W_MODEL}}" + "".join(
        f"{lbl.replace(chr(10), ' '):>{W_CONTRAST}}" for _, _, lbl in contrasts
    )
    w(c_header)
    w("-" * len(c_header))
    for m in all_models:
        row = f"{m:<{W_MODEL}}"
        for ref, cmp_c, _ in contrasts:
            s_ref = summaries[m].get(ref)
            s_cmp = summaries[m].get(cmp_c)
            if not s_ref or not s_cmp:
                row += f"{'—':>{W_CONTRAST}}"
                continue
            diff = s_cmp["harm_rate"] - s_ref["harm_rate"]
            _, p, se = _pooled_z_and_p(
                s_cmp["harm_rate"], s_cmp["total_valid"],
                s_ref["harm_rate"], s_ref["total_valid"],
            )
            row += f"{(_cell(diff, se) + _sig(p)):>{W_CONTRAST}}"
        w(row)

    # ── 3. Models-pooled contrasts (aggregate "is there an effect" test) ─────
    w("")
    w("=" * 96)
    w("  Models-pooled contrasts (sum harm/valid across all listed models)")
    w("=" * 96)
    w(f"{'Contrast (cmp − ref)':<48}"
      f"{'ref harm/n':>16}{'cmp harm/n':>16}"
      f"{'Δ harm_rate':>16}{'Z':>8}{'p':>10}")
    w("-" * 114)
    pooled_rows: list[tuple[str, float, float]] = []  # (label, diff, p)
    for ref, cmp_c, lbl in contrasts:
        h_ref, n_ref = _pooled_counts(summaries, all_models, ref)
        h_cmp, n_cmp = _pooled_counts(summaries, all_models, cmp_c)
        p_ref = h_ref / n_ref if n_ref else 0.0
        p_cmp = h_cmp / n_cmp if n_cmp else 0.0
        diff = p_cmp - p_ref
        z, p, se = _pooled_z_and_p(p_cmp, n_cmp, p_ref, n_ref)
        pooled_rows.append((lbl.replace("\n", " "), diff, p))
        w(
            f"{lbl.replace(chr(10), ' '):<48}"
            f"{f'{h_ref}/{n_ref}':>16}{f'{h_cmp}/{n_cmp}':>16}"
            f"{_cell(diff, se):>16}{z:>8.2f}{p:>9.4f}{_sig(p)}"
        )

    w("")
    w("Legend")
    w("------")
    w("Δ = cmp − ref (positive ⇒ cmp condition produces more harm)")
    w("Significance: * p<0.05  ** p<0.01  *** p<0.001  (two-tailed)")
    w("")

    # ── 3b. McNemar paired test (corrects for n-imbalance + clustering) ──────
    mcnemar_lookup: dict[str, dict] = {}
    if mcnemar_rows:
        w("=" * 96)
        w("  Paired McNemar test (scenario IDs common to all conditions)")
        w("=" * 96)
        w(f"Shared scenarios across all {len(available)} cells: {mcnemar_n_scenarios}")
        w("Pairing unit: (scenario_id, metric_id, sample_index) — same item judged under both conditions.")
        w("")
        w(f"{'Contrast (cmp − ref)':<48}"
          f"{'pairs':>10}{'n01':>8}{'n10':>8}"
          f"{'Δ harm':>12}{'stat':>10}{'p':>10}  method")
        w("-" * 114)
        for ref, cmp_c, lbl, r in mcnemar_rows:
            mcnemar_lookup[lbl.replace("\n", " ")] = r
            w(
                f"{lbl.replace(chr(10), ' '):<48}"
                f"{r['n_pairs']:>10}{r['n01']:>8}{r['n10']:>8}"
                f"{(r['delta'] * 100):>11.1f}%{r['stat']:>10.2f}{r['p']:>9.4f}{_sig(r['p'])}  {r['method']}"
            )
        w("")
        w("n01 = ref no-harm → cmp harm (cmp made it worse on that item)")
        w("n10 = ref harm   → cmp no-harm (cmp made it better on that item)")
        w("Method: exact two-sided binomial when n01+n10 < 25, else chi-square with continuity correction.")
        w("")

        # ── 3c. Plain-language summary of the McNemar analysis ───────────────
        _write_mcnemar_digest(w, all_models, mcnemar_rows, mcnemar_outcomes or {})

    # ── 4. Narrative verdict ─────────────────────────────────────────────────
    w("=" * 96)
    w("  Summary verdict")
    w("=" * 96)
    w(f"{'Contrast':<32}{'Δ (z-test)':>14}{'p (z)':>10}{'Δ (McNemar)':>16}{'p (McN)':>10}  agree?")
    w("-" * 92)
    for lbl, d, p in pooled_rows:
        mc = mcnemar_lookup.get(lbl)
        if mc:
            d_mc, p_mc = mc["delta"], mc["p"]
            sig_z, sig_mc = (not math.isnan(p) and p < 0.05), (not math.isnan(p_mc) and p_mc < 0.05)
            agree = "yes" if sig_z == sig_mc else "DIFFERS"
            w(f"{lbl:<32}{d * 100:>13.1f}%{p:>9.4f}{_sig(p):<1}"
              f"{d_mc * 100:>15.1f}%{p_mc:>9.4f}{_sig(p_mc):<1}  {agree}")
        else:
            w(f"{lbl:<32}{d * 100:>13.1f}%{p:>9.4f}{_sig(p):<1}"
              f"{'—':>16}{'—':>10}  —")
    w("")
    if mcnemar_lookup:
        sig_mc = [(lbl, mc) for lbl, mc in mcnemar_lookup.items()
                  if not math.isnan(mc["p"]) and mc["p"] < 0.05]
        if sig_mc:
            w("McNemar-significant effects (the paired test is the primary inference):")
            for lbl, mc in sig_mc:
                direction = "↑ harm" if mc["delta"] > 0 else "↓ harm"
                w(f"  - {lbl:<40}  Δ={mc['delta']:+.1%}  p={mc['p']:.4f}  ({direction})")
        else:
            w("No effects reach significance in the paired McNemar test.")
    w("")
    w("Caveats")
    w("-------")
    w("- Two tests are reported: an unpaired pooled two-proportion z-test and a paired")
    w("  McNemar test on the scenario IDs common to all four cells. The McNemar test is")
    w("  the primary inference — it respects the (scenario × sample × metric) match")
    w("  between conditions and removes the n-imbalance between 'full' and the three")
    w("  downsample cells. The z-test is retained for comparability with the per-model")
    w("  contrast table above (which is unpaired by construction).")
    w("- 'Models-pooled' rows weight models by their per-condition n. Models with more")
    w("  valid judgments dominate the pooled estimate.")
    w("- The McNemar 'pairs' counts can exceed the per-cell totals shown earlier because")
    w("  scenarios are paired item-by-item: one scenario contributes many (metric ×")
    w("  sample) pairs. Effect-size estimates (Δ) are the paired (cmp − ref) harm rate")
    w("  on the matched pairs, not a raw cell difference.")
    w("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _plot_all_conditions(
    all_models: list[str],
    summaries: dict[str, dict[str, dict | None]],
    available: list[str],
    benchmark: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Fixed semantic order: Neither → Perf-only → Adv-only → Both
    ORDERED = [
        "no-adversarial-no-perfunctory",
        "no-adversarial",
        "no-perfunctory",
        "full",
    ]
    conds    = [c for c in ORDERED if c in available]
    n_models = len(all_models)
    n_conds  = len(conds)
    bar_w    = 0.8 / n_conds
    x        = np.arange(n_models)

    all_tops_list = [
        s["harm_rate"] * 100 + _se_harm_rate(s) * 100
        for m in all_models for c in conds
        if (s := summaries[m].get(c))
    ]
    y_max = max(all_tops_list) if all_tops_list else 10.0

    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.8), 6))

    # ── Bars ──────────────────────────────────────────────────────────────────
    for i, cond in enumerate(conds):
        vals, errs = [], []
        for model_rel in all_models:
            s = summaries[model_rel].get(cond)
            vals.append((s["harm_rate"] if s else 0.0) * 100)
            errs.append((_se_harm_rate(s) if s else 0.0) * 100)

        offset = (i - (n_conds - 1) / 2) * bar_w
        ax.bar(
            x + offset, vals, bar_w,
            color=_CONDITION_COLORS[cond],
            label=_CONDITION_LABELS[cond],
            yerr=errs, capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "#555"},
        )

    # ── Significance brackets between paired bars ─────────────────────────────
    # Build a lookup: condition → bar x-offset from the group centre
    cond_offset = {c: (i - (n_conds - 1) / 2) * bar_w for i, c in enumerate(conds)}

    avail_contrasts = [
        (r, c, lbl) for r, c, lbl in _KEY_CONTRASTS
        if r in available and c in available
    ]

    step = y_max * 0.10   # vertical gap between bracket levels
    # assign a fixed vertical level to each contrast so brackets don't overlap
    bracket_level = {(r, c): lvl for lvl, (r, c, _) in enumerate(avail_contrasts)}

    for mi, model_rel in enumerate(all_models):
        for ref, cmp_c, lbl in avail_contrasts:
            s_ref = summaries[model_rel].get(ref)
            s_cmp = summaries[model_rel].get(cmp_c)
            if not s_ref or not s_cmp:
                continue
            _, p_val, _ = _pooled_z_and_p(
                s_cmp["harm_rate"], s_cmp["total_valid"],
                s_ref["harm_rate"], s_ref["total_valid"],
            )
            stars = _sig(p_val)
            if not stars:
                continue

            x_left  = x[mi] + cond_offset[ref]
            x_right = x[mi] + cond_offset[cmp_c]
            if x_left > x_right:
                x_left, x_right = x_right, x_left

            lvl    = bracket_level[(ref, cmp_c)]
            y_line = y_max * 1.04 + lvl * step
            tick   = y_max * 0.02

            ax.plot(
                [x_left, x_left, x_right, x_right],
                [y_line - tick, y_line, y_line, y_line - tick],
                lw=0.8, c="#555", solid_capstyle="butt",
            )
            short_lbl = lbl.replace("\n", " ")
            ax.text(
                (x_left + x_right) / 2, y_line,
                f"{stars} {short_lbl}",
                ha="center", va="bottom", fontsize=6.5,
                fontweight="bold", color="#333",
            )

    # ── Axes formatting ───────────────────────────────────────────────────────
    n_bracket_levels = len(avail_contrasts)
    ax.set_ylim(0, y_max * 1.04 + n_bracket_levels * step + y_max * 0.12)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(m) for m in all_models],
                       rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Combined Failure Rate (%)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.7, loc="upper left")

    fig.suptitle(
        "Design comparison — adversarial × perfunctory conditions",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        f"Benchmark: {benchmark}  |  *** p<.001  ** p<.01  * p<.05  (pooled z-test, all pairwise contrasts)",
        fontsize=9, color="#555", pad=8,
    )

    plt.tight_layout()
    out = f"all_conditions_{benchmark}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    _choices = list(_MODE_SUFFIX)
    parser = argparse.ArgumentParser(
        description="Compare any two simulation conditions on harm_rate"
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--b", "--benchmark", dest="benchmark", required=True,
        help="Benchmark folder name under results/ (e.g. emotional-dependence)",
    )
    parser.add_argument(
        "--ref", default="full", choices=_choices,
        help="Reference condition — left column (default: full = conversations/)",
    )
    parser.add_argument(
        "--cmp", default="no-adversarial", choices=_choices,
        help="Comparison condition — right column (default: no-adversarial)",
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--plot-visuals", dest="plot_visuals", action="store_true",
                        default=False, help="Save a grouped bar chart PNG")
    parser.add_argument("--all", dest="all_conditions", action="store_true",
                        default=False,
                        help="Load all conditions and plot a single grouped chart (ignores --ref/--cmp)")
    args = parser.parse_args()

    if args.all_conditions:
        compare_all(
            config_path=args.config,
            results_root=args.results_root,
            benchmark=args.benchmark,
            max_concurrency=args.concurrency,
            plot_visuals=args.plot_visuals,
        )
    else:
        compare(
            config_path=args.config,
            results_root=args.results_root,
            benchmark=args.benchmark,
            max_concurrency=args.concurrency,
            ref=args.ref,
            cmp=args.cmp,
            plot_visuals=args.plot_visuals,
        )
