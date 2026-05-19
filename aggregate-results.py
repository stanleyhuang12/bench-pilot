"""
aggregate_results.py — Combine all results_details.json files into a single
cross-model, cross-benchmark aggregate report (results_aggregate.json).

Expected directory layout (matches evaluate.py output):
    results/
      {benchmark}/
        {conv_dir}/          ← config["paths"]["conversations_dir"]
          {model_name}/
            results_details.json   ← written by evaluate.py
            results.json           ← fallback if _details file absent

Usage:
    # Auto-discover config.json in cwd
    python aggregate_results.py

    # Explicit paths
    python aggregate_results.py --config config.json --results-root results

    # Single benchmark
    python aggregate_results.py --benchmark emotional-dependency

    # Change output path
    python aggregate_results.py --output my_aggregate.json

    # Pretty-print summary to console only (no file written)
    python aggregate_results.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path



## load config or default the paths
def _load_config(config_path: str) -> dict:

    if not os.path.exists(config_path):
        return {
            "paths": {
                "conversations_dir": "conversations",
                "results_file": "results.json",
            }
        }
    with open(config_path) as f:
        return json.load(f)

## find the results_details.json
def _details_filename(results_filename: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(results_filename))
    return f"{stem}_details{ext}"

## discover all the {}_details.json files 
def _discover_detail_files(
    results_root: str,
    conv_dir: str,
    results_filename: str,
    benchmark_filter: str | None,
) -> list[dict]:
    """
    Walk results_root and return metadata dicts for every model-level directory
    that contains either a *_details.json checkpoint or a results.json with
    a "details" key.

    """
    det_fname = _details_filename(results_filename)
    found = []

    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"results-root not found: {results_root}")

    benchmarks = sorted(d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d)))
    
    if benchmark_filter:
        if benchmark_filter not in benchmarks:
            raise ValueError(
                f"Benchmark '{benchmark_filter}' not found in {results_root}. "
                f"Available: {benchmarks}"
            )
        benchmarks = [benchmark_filter]

    for bench in benchmarks:
        bench_dir  = os.path.join(results_root, bench)
        conv_base  = os.path.join(bench_dir, conv_dir)

        if not os.path.isdir(conv_base):
            conv_base = bench_dir

        # FIX: recurse so nested provider dirs (anthropic/, openai/,
        # deepinfra/google/, ...) resolve to the real model leaf, not the
        # top-level provider folder.
        det_dirs = set()
        for root, _, files in os.walk(conv_base):
            if det_fname in files or results_filename in files:
                det_dirs.add(root)
        model_dirs = sorted(det_dirs) or [conv_base]

        for mdir in model_dirs:
            det_path = os.path.join(mdir, det_fname)
            res_path = os.path.join(mdir, results_filename)

            has_det = os.path.exists(det_path)
            has_res = os.path.exists(res_path)

            if not has_det and not has_res:
                continue

            found.append({
                "benchmark": bench,
                "model": os.path.basename(mdir),
                "det_path": det_path if has_det else None,
                "res_path": res_path if has_res else None,
                "model_dir": mdir,
            })

    return found



def _load_details(entry: dict) -> list[dict]:
    if entry["det_path"] and os.path.exists(entry["det_path"]):
        with open(entry["det_path"]) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data

    if entry["res_path"] and os.path.exists(entry["res_path"]):
        with open(entry["res_path"]) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "details" in data:
            return data["details"]

    return []

BINARY = {"yes": 1, "no": 0}

def _empty() -> dict:
    return {"yes": 0, "no": 0, "fail": 0, "harm": 0, "agreements": []}

def _add_results(bucket: dict, results: list[str], harm_index: int | None) -> None:
    n = len(results)
    yes = sum(1 for r in results if r == "yes")
    no  = sum(1 for r in results if r == "no")
    bucket["yes"] += yes
    bucket["no"] += no
    bucket["fail"] += n - yes - no
    if harm_index is not None:
        bucket["harm"] += sum(
            1 for r in results if r in BINARY and BINARY[r] == harm_index
        )
    bucket["agreements"].append(max(yes, no) / n if n else 0.0)


def _finalise(bucket: dict) -> dict:
    valid = bucket["yes"] + bucket["no"]
    agrs  = bucket["agreements"]
    return {
        "yes": bucket["yes"],
        "no": bucket["no"],
        "fail": bucket["fail"],
        "harm": bucket["harm"],
        "valid": valid,
        "yes_rate": round(bucket["yes"]  / valid, 4) if valid else 0.0,
        "harm_rate": round(bucket["harm"] / valid, 4) if valid else 0.0,
        "percent_agreement": round(sum(agrs) / len(agrs), 4) if agrs else 0.0,
    }


def aggregate_all(
    entries: list[dict],
    harm_index_map: dict[str, int | None],   # metric_id → harm_index (or None)
) -> dict:
    """
    Build five aggregate views from all loaded detail records:

      by_model           — per model, rolled up across all benchmarks & metrics
      by_benchmark       — per benchmark, rolled up across all models & metrics
      by_metric          — per metric,   rolled up across all models & benchmarks
      by_scenario        — per scenario, rolled up across all models & metrics
      by_model_metric    — (model, metric_id) cross-tab
      by_model_scenario  — (model, scenario_id) cross-tab
      by_model_benchmark — (model, benchmark) cross-tab
      global             — single global summary
    """
    # Buckets keyed by dimension(s)
    g_model: dict[str, dict] = defaultdict(_empty)
    g_benchmark: dict[str, dict] = defaultdict(_empty)
    g_metric:  dict[str, dict] = defaultdict(_empty)
    g_scenario: dict[str, dict] = defaultdict(_empty)
    g_base_scenario: dict[str, dict] = defaultdict(_empty)
    g_model_metric: dict[tuple, dict] = defaultdict(_empty)
    g_model_scenario: dict[tuple, dict] = defaultdict(_empty)
    g_model_benchmark: dict[tuple, dict] = defaultdict(_empty)
    g_global: dict = _empty()

    # Extra metadata
    model_benchmarks: dict[str, set] = defaultdict(set)   # model -> {benchmarks}
    metric_names: dict[str, str]  = {}
    scenario_meta: dict[str, dict] = {}   # scenario_id -> {base_id, name}

    for entry in entries:
        bench = entry["benchmark"]
        model = entry["model"]
        details = _load_details(entry)
        model_benchmarks[model].add(bench)

        for d in details:
            mid  = d.get("metric_id", "unknown")
            sid  = d.get("scenario_id", "unknown")
            bsid  = d.get("base_scenario_id") or sid
            mname = d.get("metric_name", mid)
            results: list[str] = d.get("results", [])

            metric_names[mid] = mname
            if sid not in scenario_meta:
                scenario_meta[sid] = {"base_scenario_id": bsid}

            hi = harm_index_map.get(mid)

            _add_results(g_model[model], results, hi)
            _add_results(g_benchmark[bench], results, hi)
            _add_results(g_metric[mid], results, hi)
            _add_results(g_scenario[sid], results, hi)
            _add_results(g_base_scenario[bsid], results, hi)
            _add_results(g_model_metric[(model, mid)], results, hi)
            _add_results(g_model_scenario[(model, sid)], results, hi)
            _add_results(g_model_benchmark[(model, bench)], results, hi)
            _add_results(g_global, results, hi)

    def _finalise_dict(d: dict) -> dict:
        return {k: _finalise(v) for k, v in d.items()}

    def _finalise_tuple_dict(d: dict, keys: list[str]) -> dict:
        """Convert tuple keys to nested dicts."""
        out: dict = {}
        for tup, bucket in d.items():
            node = out
            for i, part in enumerate(tup):
                if i == len(tup) - 1:
                    node[part] = _finalise(bucket)
                else:
                    node = node.setdefault(part, {})
        return out

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": sorted(g_model.keys()),
        "benchmarks": sorted(g_benchmark.keys()),
        "global": _finalise(g_global),

        "by_model": {
            m: { **_finalise(bucket), "benchmarks": sorted(model_benchmarks[m]), }
            for m, bucket in sorted(g_model.items())
        },
        "by_benchmark": {
            k: _finalise(v)
            for k, v in sorted(g_benchmark.items())
        },
        "by_metric": {
            mid: {  **_finalise(bucket), "metric_name": metric_names.get(mid, mid), }
            for mid, bucket in sorted(g_metric.items())
        },
        "by_scenario": {
            sid: { **_finalise(bucket), **scenario_meta.get(sid, {}), }
            for sid, bucket in sorted(g_scenario.items())
        },
        "by_base_scenario": {
            k: _finalise(v)
            for k, v in sorted(g_base_scenario.items())
        },


        "by_model_metric": _finalise_tuple_dict(g_model_metric, ["model", "metric_id"]),
        "by_model_scenario":  _finalise_tuple_dict(g_model_scenario, ["model", "scenario_id"]),
        "by_model_benchmark": _finalise_tuple_dict(g_model_benchmark, ["model", "benchmark"]),
    }


def _load_harm_index_map(
    results_root: str,
    conv_dir: str,
    benchmark_filter: str | None,
    polarity_path: str = "polarity_map.json",
) -> dict[str, int | None]:
    """
    Build metric_id → harm_index (0 = positive metric, 1 = negative metric).

    Source priority:
      1. polarity_map.json (GPT-4o polarity classifier, reproduces
         finding.md Task 0: positive→0, negative→1) if present.
      2. goal.json `harm_index` per benchmark.

    Two bugs fixed vs the original:
      * test.json metrics carry no harm_index → reading them in
        unsorted os.listdir order could clobber a real goal.json value
        with None. We now read ONLY goal.json and never overwrite a
        non-None value with None.
      * (note) metric_id is not globally unique across benchmarks; the
        last writer wins. Polarity source avoids this where it agrees.
    """
    harm_map: dict[str, int | None] = {}

    # 1. polarity_map.json takes precedence (matches finding.md)
    if os.path.exists(polarity_path):
        try:
            pol = json.load(open(polarity_path))
            for _bench, metrics in pol.items():
                for mid, v in metrics.items():
                    p = v.get("polarity")
                    if p in ("positive", "negative"):
                        harm_map[mid] = 1 if p == "negative" else 0
            print(f"harm_index: loaded {len(harm_map)} metric polarities "
                  f"from {polarity_path}")
        except Exception as e:
            print(f"harm_index: failed to read {polarity_path}: {e}")

    # 2. fill gaps from goal.json only (test.json has no harm_index)
    benchmarks = sorted(
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
    ) if not benchmark_filter else [benchmark_filter]

    for bench in benchmarks:
        goal = os.path.join(results_root, bench, "goal.json")
        if not os.path.exists(goal):
            continue
        try:
            data = json.load(open(goal))
            for m in data.get("metric") or data.get("metrics") or []:
                mid = m.get("id")
                hi = m.get("harm_index")
                # never clobber an existing real value with None
                if mid and hi is not None and harm_map.get(mid) is None:
                    harm_map[mid] = hi
        except Exception:
            pass

    return harm_map


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(agg: dict) -> None:
    g = agg["global"]
    print("\n" + "=" * 62)
    print("CROSS-MODEL AGGREGATE SUMMARY")
    print("=" * 62)
    print(f"Models:  {', '.join(agg['models'])}")
    print(f"Benchmarks:  {', '.join(agg['benchmarks'])}")
    print(f"Global yes rate:  {g['yes_rate']:.1%}  ({g['yes']}/{g['valid']} yes)")
    print(f"Global harm rate: {g['harm_rate']:.1%}  ({g['harm']}/{g['valid']} harmful)")

    print("\n── By Model " + "─" * 50)
    for model, v in agg["by_model"].items():
        print(
            f"  {model:<32}  yes={v['yes_rate']:.1%}  harm={v['harm_rate']:.1%}"
            f"  ({v['yes']}/{v['valid']})  benches={v['benchmarks']}"
        )

    print("\n── By Benchmark " + "─" * 46)
    for bench, v in agg["by_benchmark"].items():
        print(
            f"  {bench:<32}  yes={v['yes_rate']:.1%}  harm={v['harm_rate']:.1%}"
            f"  ({v['yes']}/{v['valid']})"
        )

    print("\n── By Metric " + "─" * 49)
    for mid, v in agg["by_metric"].items():
        name = v.get("metric_name", mid)
        print(
            f"  {mid:<14} {name:<28}  yes={v['yes_rate']:.1%}"
            f"  harm={v['harm_rate']:.1%}  agree={v['percent_agreement']:.0%}"
        )

    print("\n── Model × Metric cross-tab " + "─" * 34)
    for model, metrics in agg["by_model_metric"].items():
        print(f"{model}")
        for mid, v in sorted(metrics.items()):
            print(
                f"{mid:<16}  yes={v['yes_rate']:.1%}"
                f" harm={v['harm_rate']:.1%}  ({v['yes']}/{v['valid']})"
            )
    print()


def run(
    config_path: str = "config.json",
    results_root: str = "results",
    benchmark_filter: str | None = None,
    output_path: str = "results_aggregate.json",
    dry_run: bool = False,
) -> dict:
    config = _load_config(config_path)
    conv_dir = config.get("paths", {}).get("conversations_dir", "conversations")
    results_file = config.get("paths", {}).get("results_file", "results.json")

    print(f"Scanning: {results_root}  (conv_dir={conv_dir})")

    entries = _discover_detail_files(
        results_root, conv_dir, results_file, benchmark_filter
    )

    if not entries:
        print("No detail files found. Run evaluate.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(entries)} (benchmark, model) pairs:")
    for e in entries:
        src = e["det_path"] or e["res_path"]
        print(f"  {e['benchmark']:30}  {e['model']:30}  ← {os.path.basename(src)}")

    harm_map = _load_harm_index_map(results_root, conv_dir, benchmark_filter)
    agg = aggregate_all(entries, harm_map)

    _print_summary(agg)

    if not dry_run:
        new_output= os.path.join(results_root, output_path) 
        with open(output_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"Saved: {output_path}  ({os.path.getsize(output_path):,} bytes)\n")
    else:
        print("(dry-run: file not written)\n")

    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate results_details.json files across models and benchmarks."
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--results-root", default="results",
        help="Root directory containing benchmark subdirectories (default: results)",
    )
    parser.add_argument(
        "--benchmark", dest="benchmark", default=None,
        help="Restrict to a single named benchmark (default: all)",
    )
    parser.add_argument(
        "--output", default="results_aggregate.json",
        help="Output JSON path (default: results_aggregate.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print summary to console without writing any file",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        results_root=args.results_root,
        benchmark_filter=args.benchmark,
        output_path=args.output,
        dry_run=args.dry_run,
    )