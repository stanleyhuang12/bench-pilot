"""
user-experience-evals.py — Phase 3 of the user-experience pipeline.

Given the matches produced by `matching-construct-to-scenario-variant.py`
(one scenario variant per construct string), run the existing bench-pilot
simulation + evaluation pipeline against EVERY target model declared in
config.json, restricted to just those variants. Emit one consolidated
result file the designer surface can render directly.

Pipeline reused from the upstream repo:
  - 2-simulation.py builds conversations (user-model ↔ target-model)
  - 3-evaluation.py scores each conversation with the evaluator model
    against the benchmark's metrics

This script is a thin orchestrator: it filters each benchmark's test.json
down to the picked variants, invokes the upstream phase scripts per
target model, then collates the per-metric results into a flat structure
keyed by (construct_string, target_model).

Usage:
    python "user experience/user-experience-evals.py" \\
        --matches "user experience/test-matching-construct-to-scenario-variant.json" \\
        --config config.json \\
        --results-root results-p \\
        --out "user experience/test-user-experience-evals.json"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_matches(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["matches"]


def load_target_models(config_path: Path) -> list[str]:
    """Read every target model from config.json (target may be a list)."""
    cfg = json.loads(config_path.read_text())
    target = cfg["models"]["target"]
    if isinstance(target, list):
        return [t["model"] for t in target]
    return [target["model"]]


def filter_test_json(src: Path, keep_ids: set[str]) -> dict:
    """Return a copy of test.json with `scenarios` filtered to keep_ids."""
    data = json.loads(src.read_text())
    data["scenarios"] = [s for s in data.get("scenarios", []) if s["id"] in keep_ids]
    return data


def run_phase(script: str, args: list[str]) -> None:
    """Invoke a numbered phase script (2-simulation.py / 3-evaluation.py)."""
    cmd = [sys.executable, str(REPO_ROOT / script), *args]
    print(f"[phase] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def collate(
    matches: list[dict],
    target_models: list[str],
    results_root: Path,
) -> dict:
    """Walk per-model results.json + results_details.json and collate."""
    out: dict = {"runs": []}
    for m in matches:
        slug = m["benchmark_slug"]
        variant_id = m["best_variant_id"]
        for tm in target_models:
            details_path = (
                results_root / slug / "conversations" / tm / "results_details.json"
            )
            summary_path = (
                results_root / slug / "conversations" / tm / "results.json"
            )

            metrics: list[dict] = []
            if details_path.exists():
                details = json.loads(details_path.read_text())
                metrics = [
                    {
                        "metric_id": d["metric_id"],
                        "metric_name": d["metric_name"],
                        "result": (d.get("results") or [None])[0],
                        "justification": (d.get("justifications") or [""])[0],
                    }
                    for d in details
                    if d.get("scenario_id") == variant_id
                ]

            summary = {}
            if summary_path.exists():
                full = json.loads(summary_path.read_text())
                summary = full.get("summary", {})

            out["runs"].append({
                "construct_string": m["construct_string"],
                "benchmark_slug": slug,
                "variant_id": variant_id,
                "target_model": tm,
                "scenario_title": m["variant"].get("title"),
                "demographic": m["variant"].get("demographic"),
                "metrics": metrics,
                "benchmark_summary": summary,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True, type=Path)
    ap.add_argument("--config", default=REPO_ROOT / "config.json", type=Path)
    ap.add_argument("--results-root", default=REPO_ROOT / "results-p", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip simulation+evaluation; just collate existing results.",
    )
    args = ap.parse_args()

    matches = load_matches(args.matches)
    target_models = load_target_models(args.config)
    print(f"[info] target models: {target_models}", flush=True)

    # Group picked variants by benchmark
    by_bench: dict[str, set[str]] = {}
    for m in matches:
        by_bench.setdefault(m["benchmark_slug"], set()).add(m["best_variant_id"])

    if not args.skip_run:
        for slug, variant_ids in by_bench.items():
            bench_dir = args.results_root / slug
            test_path = bench_dir / "test.json"
            if not test_path.exists():
                print(f"[warn] missing {test_path} — skipping {slug}", flush=True)
                continue

            # Build a filtered test.json restricted to the picked variants and
            # swap it in for the duration of the run, then restore.
            filtered = filter_test_json(test_path, variant_ids)
            with tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".json"
            ) as tf:
                json.dump(filtered, tf, indent=2)
                tmp_path = Path(tf.name)

            backup = bench_dir / "test.json.uxbak"
            shutil.copy2(test_path, backup)
            shutil.copy2(tmp_path, test_path)
            try:
                run_phase("2-simulation.py", ["--b", slug])
                run_phase("3-evaluation.py", ["--b", slug])
            finally:
                shutil.move(str(backup), str(test_path))
                tmp_path.unlink(missing_ok=True)

    payload = collate(matches, target_models, args.results_root)
    payload["target_models"] = target_models
    payload["matches_file"] = str(args.matches)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
