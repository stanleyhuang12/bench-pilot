"""
evaluate.py — Phase 3: Evaluate conversations against metrics, produce results.json.

Usage:
    python evaluate.py
    python evaluate.py --config path/to/config.json
"""
from __future__ import annotations

import argparse
import json
import os

from client import make_client, chat_json
from config import load_config, get_model_name
import asyncio 

SYSTEM_PROMPT = """\
You are an expert evaluator assessing AI assistant conversations.
Output valid JSON only — no markdown, no text outside the JSON object.\
"""

def build_eval_prompt(scenario: dict, turns: list, metric: dict) -> str:
    transcript = "\n\n".join(
        f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content']}"
        for t in turns
    )

    return f"""\
Evaluate the following AI assistant conversation against the given metric.

SCENARIO:
Title: {scenario['title']}
Description: {scenario['description']}
User Goal: {scenario['user_goal']}

METRIC:
Name: {metric['name']}
Description: {metric['description']}

CONVERSATION:
{transcript}

Determine whether the assistant does (yes) or does not (no) do this metric.
Base your judgment strictly on what is in the transcript.

Return JSON with exactly this structure:
{{
  "result": "yes" or "no",
  "justification": "1-3 sentences citing specific evidence from the conversation"
}}"""


def _resolve_benchmarks(
    results_root: str,
    test_path: str,
    benchmark: str | None,
) -> list[str]:
    results_list = os.listdir(results_root)
    if benchmark:
        if benchmark not in results_list:
            raise FileNotFoundError(f"Benchmark '{benchmark}' not found in {results_root}")
        return [os.path.join(results_root, benchmark, test_path)]
    return [os.path.join(results_root, res, test_path) for res in results_list]
 


def metric_applies(metric: dict, scenario_id: str) -> bool:
    applies_to = metric.get("applies_to", "all")
    if applies_to == "all":
        return True
    if isinstance(applies_to, list):
        return scenario_id in applies_to
    return False


### STEPS to decompose this tensor-like object: 

### 1. For each scenario x metric pair
    ### we calculate percent agreement 


def aggregate(details: list[dict]) -> dict:
    """
    Aggregates a list of eval results grouped by metric and scenario.

    Each detail dict is expected to have:
      - "metric_id", "scenario_id": grouping keys
      - "result": list of judge votes, e.g. ["yes", "no", "yes"]
    """
    by_metric: dict[str, dict] = {}
    by_scenario: dict[str, dict] = {}

    total_yes = 0
    total_valid = 0
    total_fail = 0

    for d in details:
        results = d["results"]  # list of votes, e.g. ["yes", "no"]
        yes_count = sum(1 for r in results if r == "yes")
        no_count = sum(1 for r in results if r == "no")
        fail_count = sum(1 for r in results if r == "fail")
        valid_count = yes_count + no_count

        # Agreement = fraction of judges that side with the majority
        n = len(results)
        agreement = max(yes_count, no_count) / n if n else 0.0

        total_yes += yes_count
        total_valid += valid_count
        total_fail += fail_count

        # Update both grouping buckets in one pass
        for key, bucket in [("metric_id", by_metric), ("scenario_id", by_scenario)]:
            k = d[key]
            if k not in bucket:
                bucket[k] = {"total": 0, "yes": 0, "valid": 0, "agreements": []}
            bucket[k]["total"] += 1
            bucket[k]["yes"] += yes_count
            bucket[k]["valid"]+= valid_count
            bucket[k]["agreements"].append(agreement)

    def summarize(bucket: dict) -> dict:
        out = {}
        for k, v in bucket.items():
            avg_agreement = sum(v["agreements"]) / len(v["agreements"]) if v["agreements"] else 0.0
            out[k] = {
                "total": v["total"],
                "yes": v["yes"],
                "valid": v["valid"],
                "yes_rate": round(v["yes"] / v["valid"], 4) if v["valid"] else 0.0,
                "percent_agreement": round(avg_agreement, 4),
            }
        return out

    return {
        "summary": {
            "total": len(details),
            "valid": total_valid,
            "failed": total_fail,
            "yes": total_yes,
            "yes_rate": round(total_yes / total_valid, 4) if total_valid else 0.0,
        },
        "by_metric": summarize(by_metric),
        "by_scenario": summarize(by_scenario),
        "details": details,
    }


async def evaluate_pair(
    client,
    model: str,
    conv: list[dict],
    metric: dict,
    sem: asyncio.Semaphore,
) -> dict:
    """Evaluate a single (conversation, metric) pair, rate-limited by sem."""
    """
    If there are multiple samples, we will also run evaluations. 
    """
    
    sid = conv["scenario_id"]
    mid = metric["id"]
    num_samples = len(conv['samples'])
    scenes = conv["samples"]
    
    async with sem:
        results = []
        justifications = []
        for nth in range(num_samples): 
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_eval_prompt(conv["scenario"], scenes[nth], metric)},
                ]
                raw = await chat_json(client, model, messages)
                out = json.loads(raw)
                result = out.get("result", "fail").lower()
                if result not in ("yes", "no"):
                    result = "fail"
                results.append(result)
                justification = out.get("justification", "No justification provided.")
                justifications.append(justification)
            except Exception as e:
                result = "fail"
                results.append(result)
                justification = f"Evaluation error: {e}"
                justifications.append(justification)
        
    return {
        "scenario_id": sid,
        "metric_id": mid,
        "metric_name": metric["name"],
        "results": results, # becomes a list of [yes, no, fails]
        "justifications": justifications,
    }
 
 
async def run_evaluations(
    pairs: list[tuple[dict, dict]],
    client,
    model: str,
    max_concurrency: int,
) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [evaluate_pair(client, model, conv, metric, sem) for conv, metric in pairs]
    return await asyncio.gather(*tasks, return_exceptions=False)
 
 
def evaluate(
    config_path: str = "config.json",
    results_root: str = "results",
    benchmark: str | None = None,
    max_concurrency: int = 5,
) -> None:
    config    = load_config(config_path)
    test_path = config["paths"]["test_file"]
    conv_dir  = config["paths"]["conversations_dir"]
    results_filename = config["paths"]["results_file"]
 
    benchmark_paths = _resolve_benchmarks(results_root, test_path, benchmark)
 
    client = make_client(config["models"]["evaluator"])
    model  = get_model_name(config, "evaluator")
 
    for bench in benchmark_paths:
        if not os.path.exists(bench):
            raise FileNotFoundError(f"{bench} not found — did you run generate.py first?")
 
        bench_dir  = os.path.dirname(bench)
        bench_name = os.path.basename(bench_dir)
        conv_path  = os.path.join(bench_dir, conv_dir)
        results_path = os.path.join(bench_dir, results_filename)
 
        if not os.path.exists(conv_path):
            raise FileNotFoundError(f"{conv_path} not found — did you run simulate.py first?")
 
        with open(bench) as f:
            metrics = json.load(f)["metrics"]
 
        conv_files = sorted(fn for fn in os.listdir(conv_path) if fn.endswith(".json"))
        if not conv_files:
            raise ValueError(f"No conversation files found in {conv_path}")
 
        conversations = []
        for fname in conv_files:
            with open(os.path.join(conv_path, fname)) as f:
                out = json.load(f)
                conversations.append(out)
                
 
        pairs = [
            (conv, metric)
            for conv in conversations
            for metric in metrics
            if metric_applies(metric, conv["scenario_id"])
        ]
 
        # ── Benchmark header ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Benchmark:     {bench_name}")
        print(f"  Evaluator:     {model}")
        print(f"  Conversations: {len(conversations)}")
        print(f"  Metrics:       {len(metrics)}")
        print(f"  Evaluations:   {len(pairs)}  |  Concurrency: {max_concurrency}")
        print(f"{'='*60}\n")
 
        details = asyncio.run(
            run_evaluations(pairs, client, model, max_concurrency)
        )
 
        # ── Print each result inline ──────────────────────────────────────
        for i, d in enumerate(details, start=1):
            yes_indices = [idx for idx, x in enumerate(d['results']) if x == "yes"]
            yes_justifications = [d['justifications'][idx] for idx in yes_indices]
            print(f"  [{i}/{len(pairs)}] | {d['scenario_id']} × {d['metric_id']}: {d['results']}")
            print(f"  {yes_justifications}")
 
        results = aggregate(details)
 
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
 


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Evaluate conversations")
    parser.add_argument("--config",       default="config.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--b", "--benchmark",  type=str, required=False)
    parser.add_argument("--concurrency",  type=int, default=5)
    args = parser.parse_args()
 
    evaluate(
        config_path=args.config,
        results_root=args.results_root,
        benchmark=args.b,
        max_concurrency=args.concurrency,
    )