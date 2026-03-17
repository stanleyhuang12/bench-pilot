"""
evaluate.py — Phase 3: Evaluate conversations against metrics, produce results.json.

Usage:
    python evaluate.py
    python evaluate.py --config path/to/config.json
"""

import argparse
import json
import os

from client import make_client, chat_json
from config import load_config, get_model_name


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


def evaluate(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    test_path = config["paths"]["test_file"]
    conv_dir = config["paths"]["conversations_dir"]
    results_path = config["paths"]["results_file"]

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found — run generate.py first.")
    if not os.path.exists(conv_dir):
        raise FileNotFoundError(f"{conv_dir} not found — run simulate.py first.")

    with open(test_path) as f:
        metrics = json.load(f)["metrics"]

    conv_files = sorted(f for f in os.listdir(conv_dir) if f.endswith(".json"))
    if not conv_files:
        raise ValueError(f"No conversation files found in {conv_dir}")

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

    print(f"Evaluator model: {model}")
    print(f"Conversations:   {len(conversations)}")
    print(f"Metrics:         {len(metrics)}")
    print(f"Evaluations:     {len(pairs)}\n")

    details = []
    for i, (conv, metric) in enumerate(pairs):
        sid = conv["scenario_id"]
        mid = metric["id"]
        print(f"[{i+1}/{len(pairs)}] {sid} × {mid} ({metric['name']})")

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
            justification = out.get("justification", "No justification provided.")
        except Exception as e:
            print(f"  ERROR: {e}")
            result = "fail"
            justification = f"Evaluation error: {e}"

        details.append({
            "scenario_id": sid,
            "metric_id": mid,
            "result": result,
            "justification": justification,
        })

    results = aggregate(details)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(f"\nResults:")
    print(f"  Total:     {s['total']}")
    print(f"  Passed:    {s['passed']}")
    print(f"  Pass rate: {s['pass_rate']:.1%}")
    print(f"  Saved to:  {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Evaluate conversations")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    evaluate(args.config)
