"""
run_benchmark.py — Run N benchmark trials concurrently and report consistency.

Each trial runs simulate + evaluate in its own subdirectory.
Uses the existing test.json (scenarios + metrics) for all runs.

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 3 --config config.json
"""

import argparse
import json
import os
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import load_config
from simulate import run_conversation
from client import make_client
from config import get_model_name
from evaluate import build_eval_prompt, metric_applies, aggregate, SYSTEM_PROMPT
from client import chat_json


# ---------------------------------------------------------------------------
# Per-run wrappers that redirect output paths
# ---------------------------------------------------------------------------

def run_trial(base_config: dict, run_id: int, runs_dir: str, pinpoint: bool = True) -> dict:
    """Run simulate + evaluate for one trial. Returns the results dict."""
    config = copy.deepcopy(base_config)

    trial_dir = os.path.join(runs_dir, f"run_{run_id}")
    conv_dir = os.path.join(trial_dir, "conversations")
    results_path = os.path.join(trial_dir, "results.json")

    os.makedirs(conv_dir, exist_ok=True)

    config["paths"]["conversations_dir"] = conv_dir
    config["paths"]["results_file"] = results_path

    print(f"[Run {run_id}] Starting simulation (pinpoint={pinpoint})...")
    _simulate(config, pinpoint=pinpoint)
    print(f"[Run {run_id}] Simulation done. Starting evaluation...")
    _evaluate(config)
    print(f"[Run {run_id}] Done. Pass rate: {_load_pass_rate(results_path):.1%}")

    with open(results_path) as f:
        return json.load(f)


def _simulate(config: dict, pinpoint: bool = True) -> None:
    test_path = config["paths"]["test_file"]
    conv_dir = config["paths"]["conversations_dir"]
    total_turns = config["generation"]["turns_per_conversation"]

    with open(test_path) as f:
        scenarios = json.load(f)["scenarios"]

    user_client = make_client(config["models"]["user"])
    user_model = get_model_name(config, "user")
    target_client = make_client(config["models"]["target"])
    target_model = get_model_name(config, "target")

    for scenario in scenarios:
        sid = scenario["id"]
        turns = run_conversation(
            scenario=scenario,
            user_client=user_client,
            user_model=user_model,
            target_client=target_client,
            target_model=target_model,
            total_turns=total_turns,
            pinpoint=pinpoint,
        )
        out = {"scenario_id": sid, "scenario": scenario, "turns": turns}
        with open(os.path.join(conv_dir, f"{sid}.json"), "w") as f:
            json.dump(out, f, indent=2)


def _evaluate(config: dict) -> None:
    test_path = config["paths"]["test_file"]
    conv_dir = config["paths"]["conversations_dir"]
    results_path = config["paths"]["results_file"]

    with open(test_path) as f:
        metrics = json.load(f)["metrics"]

    conv_files = sorted(f for f in os.listdir(conv_dir) if f.endswith(".json"))
    conversations = []
    for fname in conv_files:
        with open(os.path.join(conv_dir, fname)) as f:
            conversations.append(json.load(f))

    client = make_client(config["models"]["evaluator"])
    model = get_model_name(config, "evaluator")

    details = []
    for conv in conversations:
        sid = conv["scenario_id"]
        for metric in metrics:
            if not metric_applies(metric, sid):
                continue
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_eval_prompt(conv, metric)},
                ]
                raw = chat_json(client, model, messages)
                out = json.loads(raw)
                result = out.get("result", "fail").lower()
                if result not in ("pass", "fail"):
                    result = "fail"
                justification = out.get("justification", "")
            except Exception as e:
                result = "fail"
                justification = f"Error: {e}"

            details.append({
                "scenario_id": sid,
                "metric_id": metric["id"],
                "result": result,
                "justification": justification,
            })

    results = aggregate(details)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def _load_pass_rate(results_path: str) -> float:
    with open(results_path) as f:
        return json.load(f)["summary"]["pass_rate"]


# ---------------------------------------------------------------------------
# Consistency report
# ---------------------------------------------------------------------------

def consistency_report(all_results: list[dict], runs_dir: str) -> None:
    """Compare pass rates across runs and save a consistency summary."""

    n = len(all_results)
    overall_rates = [r["summary"]["pass_rate"] for r in all_results]
    mean_rate = sum(overall_rates) / n
    variance = sum((r - mean_rate) ** 2 for r in overall_rates) / n
    std_dev = variance ** 0.5

    # Collect all metric and scenario IDs
    metric_ids = list(all_results[0]["by_metric"].keys())
    scenario_ids = list(all_results[0]["by_scenario"].keys())

    by_metric = {}
    for mid in metric_ids:
        rates = [r["by_metric"].get(mid, {}).get("pass_rate", 0) for r in all_results]
        by_metric[mid] = {
            "runs": rates,
            "mean": round(sum(rates) / n, 4),
            "std_dev": round((sum((x - sum(rates)/n)**2 for x in rates) / n) ** 0.5, 4),
        }

    by_scenario = {}
    for sid in scenario_ids:
        rates = [r["by_scenario"].get(sid, {}).get("pass_rate", 0) for r in all_results]
        by_scenario[sid] = {
            "runs": rates,
            "mean": round(sum(rates) / n, 4),
            "std_dev": round((sum((x - sum(rates)/n)**2 for x in rates) / n) ** 0.5, 4),
        }

    report = {
        "num_runs": n,
        "overall": {
            "runs": overall_rates,
            "mean": round(mean_rate, 4),
            "std_dev": round(std_dev, 4),
        },
        "by_metric": by_metric,
        "by_scenario": by_scenario,
    }

    report_path = os.path.join(runs_dir, "consistency.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("CONSISTENCY REPORT")
    print("=" * 50)
    print(f"Runs:        {n}")
    print(f"Pass rates:  {[f'{r:.1%}' for r in overall_rates]}")
    print(f"Mean:        {mean_rate:.1%}")
    print(f"Std dev:     {std_dev:.1%}")
    print()
    print("By metric:")
    for mid, stats in by_metric.items():
        print(f"  {mid}: mean={stats['mean']:.1%}  std={stats['std_dev']:.1%}  runs={[f'{r:.1%}' for r in stats['runs']]}")
    print()
    print("By scenario:")
    for sid, stats in by_scenario.items():
        print(f"  {sid}: mean={stats['mean']:.1%}  std={stats['std_dev']:.1%}  runs={[f'{r:.1%}' for r in stats['runs']]}")
    print()
    print(f"Full report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run N benchmark trials concurrently")
    parser.add_argument("--runs", type=int, default=3, help="Number of trials")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=["pinpoint", "no-pinpoint"], default="pinpoint",
                        help="pinpoint: landmarks + turn counters; no-pinpoint: free-form")
    args = parser.parse_args()

    config = load_config(args.config)
    test_path = config["paths"]["test_file"]
    pinpoint = args.mode == "pinpoint"

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found — run generate.py first.")

    runs_dir = os.path.join("runs", args.mode)
    os.makedirs(runs_dir, exist_ok=True)

    print(f"Mode:    {args.mode}")
    print(f"Starting {args.runs} concurrent runs...\n")

    all_results = [None] * args.runs

    with ThreadPoolExecutor(max_workers=args.runs) as executor:
        futures = {}
        for i in range(args.runs):
            futures[executor.submit(run_trial, config, i + 1, runs_dir, pinpoint)] = i
            time.sleep(10)  # stagger starts to avoid concurrent API bursts
        for future in as_completed(futures):
            i = futures[future]
            try:
                all_results[i] = future.result()
            except Exception as e:
                print(f"[Run {i+1}] FAILED: {e}")

    all_results = [r for r in all_results if r is not None]

    if len(all_results) < 2:
        print("Not enough successful runs for a consistency report.")
        return

    consistency_report(all_results, runs_dir)


if __name__ == "__main__":
    main()
