"""
3-evaluation-batch.py — Phase 3 (Batch variant): Submit conversations to OpenAI Batch API,
poll until complete, then parse and aggregate results identically to 3-evaluation.py.

Usage:
    # Submit a new batch job
    python 3-evaluation-batch.py --b emotional-dependence --submit

    # Poll until done, then save results
    python 3-evaluation-batch.py --b emotional-dependence --retrieve

    # Submit + block until done (convenience)
    python 3-evaluation-batch.py --b emotional-dependence --submit --wait

    # Re-aggregate from already-parsed details (no API calls)
    python 3-evaluation-batch.py --b emotional-dependence --aggregate

Batch artifacts are stored in:
    results/<benchmark>/batch/<model_name>/
        batch_meta.json          — batch_id + request-id → (scenario_id, sample_idx, metric_id) map
        requests.jsonl           — the .jsonl file uploaded to OpenAI
        results.jsonl            — raw output downloaded from OpenAI
"""
from __future__ import annotations

import argparse
import json
import os
import time
import importlib
import importlib.util
import warnings

import openai


# 3-evaluation.py has a hyphen in its name, so we load it via importlib
_eval_mod = importlib.util.spec_from_file_location(
    "evaluation",
    os.path.join(os.path.dirname(__file__), "3-evaluation.py"),
)
_eval = importlib.util.module_from_spec(_eval_mod)
_eval_mod.loader.exec_module(_eval)

SYSTEM_PROMPT      = _eval.SYSTEM_PROMPT
aggregate          = _eval.aggregate
build_eval_prompt  = _eval.build_eval_prompt
_resolve_benchmarks = _eval._resolve_benchmarks
_get_model_dirs    = _eval._get_model_dirs
_details_path      = _eval._details_path
_save_details      = _eval._save_details
_save_results      = _eval._save_results
_load_details      = _eval._load_details


# ---------------------------------------------------------------------------
# Recursive model-dir discovery (handles 1, 2, or 3-level nesting)
# ---------------------------------------------------------------------------

def _get_leaf_model_dirs(conv_base: str) -> list[tuple[str, str]]:
    """
    Return (abs_path, rel_name) for every directory under conv_base that
    contains at least one scenario_*.json file.
    rel_name preserves provider nesting, e.g. 'anthropic/claude-sonnet-4-6'.
    """
    results = []
    for dirpath, dirnames, filenames in os.walk(conv_base):
        dirnames.sort()
        if any(f.startswith("scenario_") and f.endswith(".json") for f in filenames):
            rel = os.path.relpath(dirpath, conv_base)
            results.append((dirpath, rel))
    return results


# ---------------------------------------------------------------------------
# Batch directory helpers
# ---------------------------------------------------------------------------

def _batch_dir(bench_dir: str, model_name: str) -> str:
    d = os.path.join(bench_dir, "batch", model_name)
    os.makedirs(d, exist_ok=True)
    return d


def _batch_meta_path(batch_dir: str) -> str:
    return os.path.join(batch_dir, "batch_meta.json")


def _requests_path(batch_dir: str) -> str:
    return os.path.join(batch_dir, "requests.jsonl")


def _results_path(batch_dir: str) -> str:
    return os.path.join(batch_dir, "results.jsonl")


# ---------------------------------------------------------------------------
# Build JSONL requests
# ---------------------------------------------------------------------------

def _custom_id(scenario_id: str, sample_idx: int) -> str:
    return f"{scenario_id}__s{sample_idx}"


def build_requests(
    conversations: list[dict],
    metrics: list[dict],
    model: str,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Returns:
        requests  — list of OpenAI batch request objects
        index_map — custom_id → {scenario_id, sample_idx, base_scenario_id}
    """
    requests = []
    index_map: dict[str, dict] = {}

    for conv in conversations:
        sid = conv["scenario_id"]
        base_sid = conv["scenario"].get("base_scenario_id")
        for i, turns in enumerate(conv["samples"]):
            cid = _custom_id(sid, i)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_eval_prompt(conv["scenario"], turns, metrics)},
            ]
            requests.append({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                },
            })
            index_map[cid] = {
                "scenario_id": sid,
                "sample_idx": i,
                "base_scenario_id": base_sid,
            }

    return requests, index_map


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit(
    bench_dir: str,
    model_dir: str,
    model_name: str,
    metrics: list[dict],
    results_filename: str,
    evaluator_model: str,
    openai_api_key: str,
) -> str:
    client = openai.OpenAI(api_key=openai_api_key)

    results_path = os.path.join(model_dir, results_filename)
    det_path = _details_path(model_dir, results_filename)
    bd = _batch_dir(bench_dir, model_name)
    meta_path = _batch_meta_path(bd)

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Batch already submitted: {meta['batch_id']}  (delete {meta_path} to resubmit)")
        return meta["batch_id"]

    if os.path.exists(results_path):
        warnings.warn(
            f"'{results_path}' already exists. "
            "Pass --aggregate to re-aggregate, or delete it to re-run."
        )

    conv_files = sorted(
        fn for fn in os.listdir(model_dir)
        if fn.endswith(".json") and fn not in {
            os.path.basename(results_path),
            os.path.basename(det_path),
            "cost.json",
            "cost_checkpoint.json",
        }
        and not fn.startswith("cost")
    )
    if not conv_files:
        print(f"No conversation files in {model_dir}. Skipping.")
        return ""

    conversations = []
    for fname in conv_files:
        with open(os.path.join(model_dir, fname)) as f:
            conversations.append(json.load(f))

    requests, index_map = build_requests(conversations, metrics, evaluator_model)
    total = len(requests)

    # Write requests.jsonl
    req_path = _requests_path(bd)
    with open(req_path, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    print(f"Uploading {total} requests for {model_name}...")
    with open(req_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    meta = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "model_name": model_name,
        "evaluator_model": evaluator_model,
        "num_requests": total,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index_map": index_map,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Submitted batch {batch.id}  ({total} requests)")
    print(f"Batch meta: {meta_path}")
    return batch.id


# ---------------------------------------------------------------------------
# Poll / retrieve
# ---------------------------------------------------------------------------

def _poll_until_done(client: openai.OpenAI, batch_id: str, poll_interval: int = 30) -> object:
    print(f"Polling batch {batch_id}...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        print(
            f"  status={status}  "
            f"completed={counts.completed}  failed={counts.failed}  total={counts.total}"
        )
        if status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(poll_interval)


def retrieve(
    bench_dir: str,
    model_name: str,
    model_dir: str,
    metrics: list[dict],
    results_filename: str,
    openai_api_key: str,
    wait: bool = False,
) -> None:
    client = openai.OpenAI(api_key=openai_api_key)
    bd = _batch_dir(bench_dir, model_name)
    meta_path = _batch_meta_path(bd)

    if not os.path.exists(meta_path):
        print(f"No batch meta found at {meta_path}. Run --submit first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    batch_id = meta["batch_id"]
    index_map: dict[str, dict] = meta["index_map"]

    if wait:
        batch = _poll_until_done(client, batch_id)
    else:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id} status: {batch.status}")
        if batch.status != "completed":
            print("Not complete yet. Re-run with --retrieve (or --submit --wait to block).")
            return

    if batch.status != "completed":
        print(f"Batch ended with status '{batch.status}'. Cannot parse results.")
        return

    # Download results.jsonl
    raw_content = client.files.content(batch.output_file_id).content
    res_path = _results_path(bd)
    with open(res_path, "wb") as f:
        f.write(raw_content)
    print(f"Downloaded results to {res_path}")

    # Parse results into the same detail format as 3-evaluation.py
    metric_lookup = {m["id"]: m.get("name", m["id"]) for m in metrics}
    list_mid = [m["id"] for m in metrics]

    # Accumulate per-(scenario_id, metric_id) across samples
    # structure: acc[(sid, mid)] = {"results": [], "justifications": [], ...}
    acc: dict[tuple, dict] = {}

    with open(res_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row["custom_id"]
            info = index_map.get(cid)
            if info is None:
                print(f"Warning: unknown custom_id {cid}, skipping.")
                continue

            sid = info["scenario_id"]
            base_sid = info["base_scenario_id"]

            # Parse the model output
            try:
                body = row["response"]["body"]
                content = body["choices"][0]["message"]["content"]
                out = json.loads(content)
                for res_dict in out.values():
                    result = res_dict.get("result", "fail").lower()
                    res_dict["result"] = result if result in ("yes", "no") else "fail"
                    res_dict["justification"] = res_dict.get("justification", "No justification provided.")
            except Exception as e:
                print(f"Parse error for {cid}: {e} — marking all metrics as fail")
                out = {
                    mid: {"result": "fail", "justification": f"Parse error: {e}"}
                    for mid in list_mid
                }

            for mid in list_mid:
                res = out.get(mid, {}).get("result", "fail")
                js = out.get(mid, {}).get("justification", "No justification provided.")
                key = (sid, mid)
                if key not in acc:
                    acc[key] = {
                        "scenario_id": sid,
                        "base_scenario_id": base_sid,
                        "metric_id": mid,
                        "metric_name": metric_lookup.get(mid, mid),
                        "num_samples": 0,
                        "results": [],
                        "justifications": [],
                    }
                acc[key]["results"].append(res)
                acc[key]["justifications"].append(js)
                acc[key]["num_samples"] += 1

    details = list(acc.values())

    results_path = os.path.join(model_dir, results_filename)
    det_path = _details_path(model_dir, results_filename)

    _save_details(det_path, details)
    results = aggregate(details, metrics)
    _save_results(results_path, results)

    s = results["summary"]
    print(f"\n{model_name} complete")
    print(f"Yes rate:  {s['yes_rate']:.2%}")
    print(f"Harm rate: {s['harm_rate']:.2%}")
    print(f"Details:   {det_path}")
    print(f"Results:   {results_path}")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_submit(
    results_root: str,
    benchmark: str | None,
    evaluator_model: str,
    openai_api_key: str,
    test_path: str = "test.json",
    conv_dir: str = "conversations",
    results_filename: str = "results.json",
) -> None:
    benchmark_paths = _resolve_benchmarks(results_root, test_path, benchmark)

    for bench_path in benchmark_paths:
        if not os.path.exists(bench_path):
            print(f"\nSkipping — test file not found: {bench_path}")
            continue

        bench_dir = os.path.dirname(bench_path)
        bench_name = os.path.basename(bench_dir)
        conv_base = os.path.join(bench_dir, conv_dir)

        with open(bench_path) as f:
            test_data = json.load(f)
        metrics: list[dict] = test_data.get("metrics") or test_data.get("metric")

        leaf_dirs = _get_leaf_model_dirs(conv_base)
        if not leaf_dirs:
            print(f"No model dirs in {conv_base}. Skipping.")
            continue

        for model_dir, model_name in leaf_dirs:
            print(f"\n{'='*62}")
            print(f"Benchmark: {bench_name}  |  Model: {model_name}")
            print(f"Evaluator: {evaluator_model}")
            print(f"{'='*62}")
            submit(
                bench_dir=bench_dir,
                model_dir=model_dir,
                model_name=model_name,
                metrics=metrics,
                results_filename=results_filename,
                evaluator_model=evaluator_model,
                openai_api_key=openai_api_key,
            )


def run_retrieve(
    results_root: str,
    benchmark: str | None,
    openai_api_key: str,
    wait: bool,
    test_path: str = "test.json",
    conv_dir: str = "conversations",
    results_filename: str = "results.json",
) -> None:
    benchmark_paths = _resolve_benchmarks(results_root, test_path, benchmark)

    for bench_path in benchmark_paths:
        if not os.path.exists(bench_path):
            print(f"\nSkipping — test file not found: {bench_path}")
            continue

        bench_dir = os.path.dirname(bench_path)
        bench_name = os.path.basename(bench_dir)
        conv_base = os.path.join(bench_dir, conv_dir)

        with open(bench_path) as f:
            test_data = json.load(f)
        metrics: list[dict] = test_data.get("metrics") or test_data.get("metric")

        leaf_dirs = _get_leaf_model_dirs(conv_base)
        if not leaf_dirs:
            print(f"No model dirs in {conv_base}. Skipping.")
            continue

        for model_dir, model_name in leaf_dirs:
            print(f"\n{'='*62}")
            print(f"Benchmark: {bench_name}  |  Model: {model_name}")
            print(f"{'='*62}")
            retrieve(
                bench_dir=bench_dir,
                model_name=model_name,
                model_dir=model_dir,
                metrics=metrics,
                results_filename=results_filename,
                openai_api_key=openai_api_key,
                wait=wait,
            )


def run_aggregate_only(
    results_root: str,
    benchmark: str | None,
    test_path: str = "test.json",
    conv_dir: str = "conversations",
    results_filename: str = "results.json",
) -> None:
    """Re-aggregate from details checkpoint, same as 3-evaluation.py --aggregate."""
    benchmark_paths = _resolve_benchmarks(results_root, test_path, benchmark)

    for bench_path in benchmark_paths:
        if not os.path.exists(bench_path):
            print(f"\nSkipping — test file not found: {bench_path}")
            continue

        bench_dir = os.path.dirname(bench_path)
        bench_name = os.path.basename(bench_dir)
        conv_base = os.path.join(bench_dir, conv_dir)

        with open(bench_path) as f:
            test_data = json.load(f)
        metrics: list[dict] = test_data.get("metrics") or test_data.get("metric")

        leaf_dirs = _get_leaf_model_dirs(conv_base)
        if not leaf_dirs:
            print(f"No model dirs in {conv_base}. Skipping.")
            continue

        for model_dir, model_name in leaf_dirs:
            results_path = os.path.join(model_dir, results_filename)
            det_path = _details_path(model_dir, results_filename)

            print(f"\nModel: {model_name}")
            try:
                details = _load_details(det_path, results_path)
            except FileNotFoundError as e:
                print(f"Skipping — {e}")
                continue

            results = aggregate(details, metrics)
            _save_results(results_path, results)
            s = results["summary"]
            print(f"Yes rate: {s['yes_rate']:.2%}  |  Harm rate: {s['harm_rate']:.2%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 (Batch): Evaluate via OpenAI Batch API")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--b", "--benchmark", dest="benchmark", type=str, required=False)
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Evaluator model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (defaults to $OPENAI_API_KEY)",
    )
    parser.add_argument("--submit", action="store_true", help="Upload and submit batch jobs")
    parser.add_argument("--retrieve", action="store_true", help="Download and parse completed batches")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="When retrieving, poll until the batch is complete (use with --retrieve or --submit)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Re-aggregate from details checkpoint without re-running any API calls",
    )
    args = parser.parse_args()

    if args.aggregate:
        run_aggregate_only(
            results_root=args.results_root,
            benchmark=args.benchmark,
        )
    elif args.submit and args.retrieve:
        run_submit(
            results_root=args.results_root,
            benchmark=args.benchmark,
            evaluator_model=args.model,
            openai_api_key=args.api_key,
        )
        run_retrieve(
            results_root=args.results_root,
            benchmark=args.benchmark,
            openai_api_key=args.api_key,
            wait=True,
        )
    elif args.submit:
        run_submit(
            results_root=args.results_root,
            benchmark=args.benchmark,
            evaluator_model=args.model,
            openai_api_key=args.api_key,
        )
    elif args.retrieve:
        run_retrieve(
            results_root=args.results_root,
            benchmark=args.benchmark,
            openai_api_key=args.api_key,
            wait=args.wait,
        )
    else:
        parser.print_help()
