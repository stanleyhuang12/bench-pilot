"""
run_factorial_ablation.py — One-shot orchestrator for the 2×2 (adversarial × perfunctory) study.

For each benchmark, this script:
  1. Generates test_no_adversarial.json via 1-test-scenario-distilled.py if it
     doesn't already exist (skip with --regenerate to force).
  2. Picks a deterministic, metric-stratified scenario subset of size --n
     that exists in BOTH test.json and test_no_adversarial.json.
  3. Runs all four 2×2 cells through 2-simulation.py with that same subset,
     so cells pair 1:1 by scenario_id for paired analysis later. Cell A
     (adversarial=ON, perfunctory=ON) is the canonical BASELINE — it matches
     the flags used in the main full-scale run — and lives in the bare
     `downsample/` directory. The other three cells append a `-no-<factor>`
     tag per OFF factor, so the suffix encodes deviations from baseline:

         A: adversarial=ON,  perfunctory=ON   → conversations/downsample/
         B: adversarial=ON,  perfunctory=OFF  → conversations/downsample-no-perfunctory/
         C: adversarial=OFF, perfunctory=ON   → conversations/downsample-no-adversarial/
         D: adversarial=OFF, perfunctory=OFF  → conversations/downsample-no-adversarial-no-perfunctory/

  4. Writes a manifest JSON pointing compare_realism_injection.py at every
     conversation directory it produced.

Usage:
    python3 run_factorial_ablation.py
    python3 run_factorial_ablation.py --benchmarks emotional-dependence
    python3 run_factorial_ablation.py --n 30 --ns 1 --semaphore 8
    python3 run_factorial_ablation.py --skip-cells A_adv_perf,B_adv_noperf
    python3 run_factorial_ablation.py --dry-run   # print every command, run nothing
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Defaults — override on the command line if needed.
DEFAULT_BENCHMARKS  = ["emotional-dependence", "humanebench"]
DEFAULT_CONFIG      = "config-test-adversarial-subset.json"
DEFAULT_RESULTS_ROOT = "results"
DEFAULT_N           = 45
DEFAULT_SAMPLES     = 1
DEFAULT_SEMAPHORE   = 5
DEFAULT_SEED        = 8913

# (label, no_adversarial, perfunctory, test_file)
CELLS = [
    ("A_adv_perf",     False, True,  "test.json"),
    ("B_adv_noperf",   False, False, "test.json"),
    ("C_noadv_perf",   True,  True,  "test_no_adversarial.json"),
    ("D_noadv_noperf", True,  False, "test_no_adversarial.json"),
]


def _shell(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _run(cmd: list[str], label: str, dry_run: bool, capture: bool = False) -> tuple[int, str, str]:
    print(f"\n──── {label} ────")
    print(f"$ {_shell(cmd)}")
    if dry_run:
        return 0, "", ""
    t0 = time.perf_counter()
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        stdout, stderr = r.stdout, r.stderr
    else:
        r = subprocess.run(cmd)
        stdout, stderr = "", ""
    elapsed = time.perf_counter() - t0
    print(f"  → exit {r.returncode}  ({elapsed:.1f}s)")
    return r.returncode, stdout, stderr


def phase1_generate(benchmark: str, results_root: str, regenerate: bool, dry_run: bool) -> None:
    out = Path(results_root) / benchmark / "test_no_adversarial.json"
    if out.exists() and not regenerate:
        print(f"\n──── Phase 1: {benchmark} ────")
        print(f"  ✓ {out} already exists — skipping (pass --regenerate to force)")
        return
    cmd = [
        sys.executable, "1-test-scenario-distilled.py",
        "--b", benchmark,
        "--no-adversarial-scenarios",
        "--demographic-base", "age",
        "--demographic-random", "gender",
        "--results-root", results_root,
    ]
    if regenerate and out.exists():
        cmd.append("--overwrite")
    rc, _, _ = _run(cmd, f"Phase 1: generate test_no_adversarial.json for {benchmark}", dry_run)
    if rc != 0:
        sys.exit(f"FAILED Phase 1 for {benchmark}")


def phase1_pick(benchmark: str, n: int, seed: int, results_root: str, dry_run: bool) -> str:
    test_path = Path(results_root) / benchmark / "test.json"
    no_adv_path = Path(results_root) / benchmark / "test_no_adversarial.json"
    if not test_path.exists():
        sys.exit(f"FAILED to pick subset: {test_path} not found")
    if not no_adv_path.exists():
        if dry_run:
            return "<scenario-ids-placeholder>"
        sys.exit(f"FAILED to pick subset: {no_adv_path} not found")

    cmd = [
        sys.executable, "pick_scenario_subset.py",
        str(test_path), str(n),
        "--seed", str(seed),
        "--also-in", str(no_adv_path),
    ]
    rc, stdout, stderr = _run(cmd, f"Phase 1: pick {n} scenarios for {benchmark}", dry_run, capture=True)
    if rc != 0:
        sys.exit(f"FAILED pick_scenario_subset for {benchmark}:\n{stderr}")
    if dry_run:
        return "<scenario-ids-placeholder>"
    if stderr.strip():
        print(stderr.rstrip())
    ids = stdout.strip()
    if not ids:
        sys.exit(f"FAILED pick_scenario_subset for {benchmark}: empty output")
    return ids


def phase2_cell(
    benchmark: str,
    label: str,
    no_adv: bool,
    perf: bool,
    test_file: str,
    scenario_ids: str,
    config: str,
    num_samples: int,
    semaphore: int,
    results_root: str,
    dry_run: bool,
) -> Path:
    """Run one 2×2 cell. Returns the conversations directory the cell wrote to."""
    cmd = [
        sys.executable, "2-simulation.py",
        "--b", benchmark,
        "--config", config,
        "--results-root", results_root,
        "--downsample",
        "--scenario-ids", scenario_ids,
        "--ns", str(num_samples),
        "--semaphore", str(semaphore),
        "--test-file", test_file,
    ]
    if no_adv:
        cmd.append("--no-adversarial")
    if perf:
        cmd.append("--p")
    rc, _, _ = _run(cmd, f"Phase 2 cell {label}: {benchmark}", dry_run)
    if rc != 0:
        sys.exit(f"FAILED cell {label} for {benchmark}")

    # Mirror 2-simulation.py's simulate_downsample suffix logic so we know
    # where the conversations landed. (Kept in sync with that function.)
    # Cell A (adv=ON, perf=ON) → bare "downsample/"; each OFF factor appends
    # a "-no-<factor>" tag.
    parts = ["downsample"]
    if no_adv:
        parts.append("no-adversarial")
    if not perf:
        parts.append("no-perfunctory")
    suffix = "-".join(parts)
    return Path(results_root) / benchmark / "conversations" / suffix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end 2×2 (adversarial × perfunctory) ablation orchestrator."
    )
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        help=f"Default: {DEFAULT_BENCHMARKS}")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Default: {DEFAULT_CONFIG}")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Scenarios per benchmark to simulate (default {DEFAULT_N})")
    parser.add_argument("--ns", "--num-samples", dest="num_samples",
                        type=int, default=DEFAULT_SAMPLES,
                        help=f"Conversation samples per scenario (default {DEFAULT_SAMPLES})")
    parser.add_argument("--semaphore", type=int, default=DEFAULT_SEMAPHORE,
                        help=f"Per-cell concurrency (default {DEFAULT_SEMAPHORE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Subset-picking seed (default {DEFAULT_SEED})")
    parser.add_argument("--skip-cells", default="",
                        help="Comma-separated cell labels to skip: "
                             "A_adv_perf, B_adv_noperf, C_noadv_perf, D_noadv_noperf")
    parser.add_argument("--regenerate", action="store_true",
                        help="Overwrite an existing test_no_adversarial.json instead of skipping")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command but execute nothing")
    parser.add_argument("--manifest", default="factorial_ablation_manifest.json",
                        help="Where to write the run manifest (consumed by compare_realism_injection.py)")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip_cells.split(",") if s.strip()}
    unknown = skip - {label for label, *_ in CELLS}
    if unknown:
        sys.exit(f"Unknown --skip-cells: {unknown}. "
                 f"Valid: {[c[0] for c in CELLS]}")

    print("=" * 72)
    print("2×2 factorial ablation — adversarial × perfunctory")
    print("=" * 72)
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Config:     {args.config}")
    print(f"Subset:     n={args.n}, seed={args.seed}")
    print(f"Samples:    ns={args.num_samples}, semaphore={args.semaphore}")
    if skip:
        print(f"Skipping cells: {sorted(skip)}")
    if args.dry_run:
        print("DRY RUN — no commands will execute")
    print()

    manifest: dict = {
        "config": args.config,
        "n": args.n,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "benchmarks": {},
    }

    for benchmark in args.benchmarks:
        print("\n" + "█" * 72)
        print(f"█  Benchmark: {benchmark}")
        print("█" * 72)

        phase1_generate(benchmark, args.results_root, args.regenerate, args.dry_run)
        scenario_ids = phase1_pick(benchmark, args.n, args.seed, args.results_root, args.dry_run)

        manifest["benchmarks"][benchmark] = {
            "scenario_ids": scenario_ids.split(",") if scenario_ids and scenario_ids != "<scenario-ids-placeholder>" else [],
            "cells": {},
        }

        for label, no_adv, perf, test_file in CELLS:
            if label in skip:
                print(f"\n──── skip cell {label} for {benchmark} ────")
                continue
            conv_dir = phase2_cell(
                benchmark, label, no_adv, perf, test_file, scenario_ids,
                args.config, args.num_samples, args.semaphore,
                args.results_root, args.dry_run,
            )
            manifest["benchmarks"][benchmark]["cells"][label] = {
                "adversarial": not no_adv,
                "perfunctory": perf,
                "test_file": test_file,
                "conversations_dir": str(conv_dir),
            }

    if not args.dry_run:
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n✓ Manifest written to {args.manifest}")
    else:
        print(f"\n(dry-run) Manifest would be written to {args.manifest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
