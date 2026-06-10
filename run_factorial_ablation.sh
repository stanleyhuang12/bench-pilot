#!/usr/bin/env bash
# run_factorial_ablation.sh — Execute the 2×2 (adversarial × perfunctory) ablation.
#
# Cell A (adv=ON, perf=ON) is the canonical baseline; it shares its flags with
# the main full-scale run and lands in conversations/downsample/.
#
# By default this script runs only the THREE non-baseline cells (B, C, D) — the
# baseline is already paired conceptually with the main full-scale run. Drop
# --skip-cells below to also generate the downsample-scale version of Cell A.
#
# Usage:
#   ./run_factorial_ablation.sh                       # B, C, D on both benchmarks
#   ./run_factorial_ablation.sh --benchmarks emotional-dependence
#   BENCHMARKS="humanebench" ./run_factorial_ablation.sh
#
set -euo pipefail

cd "$(dirname "$0")"

N=45
NS=1
SEMAPHORE=50
SEED=8913

python3 run_factorial_ablation.py \
    --n "${N}" \
    --ns "${NS}" \
    --semaphore "${SEMAPHORE}" \
    --seed "${SEED}" \
    --skip-cells A_adv_perf \
    "$@"

echo
echo "✓ Done. Cells written to:"
echo "    results/<benchmark>/conversations/downsample-no-perfunctory/             (B)"
echo "    results/<benchmark>/conversations/downsample-no-adversarial/             (C)"
echo "    results/<benchmark>/conversations/downsample-no-adversarial-no-perfunctory/ (D)"
echo "Manifest: factorial_ablation_manifest.json"
