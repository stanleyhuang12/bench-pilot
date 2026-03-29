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


def build_eval_prompt(conversation: dict, metric: dict) -> str:
    scenario = conversation["scenario"]

    transcript = "\n\n".join(
        f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content']}"
        for t in conversation["turns"]
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

Determine whether the assistant PASSES or FAILS this metric.
Base your judgment strictly on what is in the transcript.

Return JSON with exactly this structure:
{{
  "result": "pass" or "fail",
  "justification": "1-3 sentences citing specific evidence from the conversation"
}}\
"""

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


def aggregate(details: list[dict]) -> dict:
    total = len(details)
    passed = sum(1 for d in details if d["result"] == "pass")

    by_metric: dict[str, dict] = {}
    by_scenario: dict[str, dict] = {}

    for d in details:
        for key, bucket in [("metric_id", by_metric), ("scenario_id", by_scenario)]:
            k = d[key]
            if k not in bucket:
                bucket[k] = {"total": 0, "passed": 0}
            bucket[k]["total"] += 1
            if d["result"] == "pass":
                bucket[k]["passed"] += 1

    def pass_rate(b: dict) -> dict:
        return {k: {**v, "pass_rate": round(v["passed"] / v["total"], 4)} for k, v in b.items()}

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        },
        "by_metric": pass_rate(by_metric),
        "by_scenario": pass_rate(by_scenario),
        "details": details,
    }

async def evaluate_pair(
    client,
    model: str,
    conv: dict,
    metric: dict,
    sem: asyncio.Semaphore,
) -> dict:
    """Evaluate a single (conversation, metric) pair, rate-limited by sem."""
    sid = conv["scenario_id"]
    mid = metric["id"]
 
    async with sem:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_eval_prompt(conv, metric)},
            ]
            raw = await chat_json(client, model, messages)
            out = json.loads(raw)
            result = out.get("result", "fail").lower()
            if result not in ("pass", "fail"):
                result = "fail"
            justification = out.get("justification", "No justification provided.")
        except Exception as e:
            result = "fail"
            justification = f"Evaluation error: {e}"
        
 
    return {
        "scenario_id":   sid,
        "metric_id":     mid,
        "metric_name":   metric["name"],
        "result":        result,
        "justification": justification,
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
                conversations.append(json.load(f))
 
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
            icon = "✓" if d["result"] == "pass" else "✗"
            print(f"  [{i}/{len(pairs)}] {icon} {d['scenario_id']} × {d['metric_id']}")
            print(f"  {d['justification']}")
 
        results = aggregate(details)
 
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
 


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Evaluate conversations")
    parser.add_argument("--config",       default="config.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("b", "--benchmark",    type=str, required=False)
    parser.add_argument("--concurrency",  type=int, default=5)
    args = parser.parse_args()
 
    evaluate(
        config_path=args.config,
        results_root=args.results_root,
        benchmark=args.benchmark,
        max_concurrency=args.concurrency,
    )