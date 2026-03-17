"""
eval_consistency.py — Test evaluator determinism by running the same
conversation × metric pair N times and measuring verdict stability.

Usage:
    python eval_consistency.py --runs 5
    python eval_consistency.py --runs 5 --config config.json --conv-dir runs/pinpoint/run_1/conversations
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from client import make_client, chat_json
from config import load_config, get_model_name
from evaluate import build_eval_prompt, SYSTEM_PROMPT, metric_applies


def evaluate_once(client, model, conv, metric):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_eval_prompt(conv, metric)},
    ]
    raw = chat_json(client, model, messages)
    out = json.loads(raw)
    result = out.get("result", "fail").lower()
    if result not in ("pass", "fail"):
        result = "fail"
    return result, out.get("justification", "")


def run(config_path, conv_dir, n_runs):
    config = load_config(config_path)
    test_path = config["paths"]["test_file"]

    with open(test_path) as f:
        metrics = json.load(f)["metrics"]

    conv_files = sorted(f for f in os.listdir(conv_dir) if f.endswith(".json"))
    conversations = []
    for fname in conv_files:
        with open(os.path.join(conv_dir, fname)) as f:
            conversations.append(json.load(f))

    client = make_client(config["models"]["evaluator"])
    model = get_model_name(config, "evaluator")

    pairs = [
        (conv, metric)
        for conv in conversations
        for metric in metrics
        if metric_applies(metric, conv["scenario_id"])
    ]

    print(f"Evaluator: {model}")
    print(f"Pairs: {len(pairs)}  ×  {n_runs} runs each  =  {len(pairs)*n_runs} total calls")
    print()

    # For each pair, run N times and collect verdicts
    results = {}  # (scenario_id, metric_id) -> list of verdicts

    for conv, metric in pairs:
        key = (conv["scenario_id"], metric["id"])
        print(f"  {key[0]} × {key[1]} — running {n_runs}x ...", end=" ", flush=True)

        verdicts = []
        for _ in range(n_runs):
            result, _ = evaluate_once(client, model, conv, metric)
            verdicts.append(result)

        results[key] = verdicts
        agreement = len(set(verdicts)) == 1
        print(f"  {verdicts}  {'STABLE' if agreement else 'FLIP'}")

    # Summary
    print()
    print("=" * 60)
    print("EVALUATOR CONSISTENCY REPORT")
    print("=" * 60)

    stable = 0
    flipped = []
    for key, verdicts in results.items():
        if len(set(verdicts)) == 1:
            stable += 1
        else:
            flipped.append((key, verdicts))

    total = len(results)
    print(f"Stable (same verdict all {n_runs} runs): {stable}/{total} ({stable/total*100:.0f}%)")
    print(f"Flipped (verdict changed):               {len(flipped)}/{total} ({len(flipped)/total*100:.0f}%)")

    if flipped:
        print()
        print("Items that flipped:")
        for (scen, met), verdicts in flipped:
            passes = verdicts.count("pass")
            fails = verdicts.count("fail")
            print(f"  {scen} × {met}: {verdicts}  (pass {passes}x, fail {fails}x)")

    # Per-item flip rate breakdown
    print()
    print("Per-item verdict distribution:")
    for (scen, met), verdicts in sorted(results.items()):
        passes = verdicts.count("pass")
        marker = "" if len(set(verdicts)) == 1 else "  <-- FLIP"
        print(f"  {scen} × {met}: pass={passes}/{n_runs}  fail={n_runs-passes}/{n_runs}{marker}")

    # Save
    out = {
        "evaluator": model,
        "conv_dir": conv_dir,
        "n_runs": n_runs,
        "total_pairs": total,
        "stable": stable,
        "flipped": len(flipped),
        "stability_rate": round(stable / total, 4),
        "details": {
            f"{k[0]}x{k[1]}": v for k, v in results.items()
        }
    }
    out_path = "runs/eval_consistency.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--conv-dir", default="runs/pinpoint/run_1/conversations")
    args = parser.parse_args()
    run(args.config, args.conv_dir, args.runs)
