#!/usr/bin/env bash
# aggregate_costs_filtered.sh — Summarize cost.json files across benchmarks, with optional step filtering.
# Compatible with macOS Bash 3.x

set -euo pipefail
trap 'echo "[ERROR] line $LINENO — $BASH_COMMAND" >&2' ERR

RESULTS_ROOT="results-p"
FILTER_STEPS=()  # Empty means all steps

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --steps)
            IFS=',' read -r -a FILTER_STEPS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# All possible steps
ALL_STEPS=("goal_normalization" "base_scenario_construction" "demographic_scenario_construction" "simulation" "evaluation")

# Determine which steps to use
if [ ${#FILTER_STEPS[@]} -eq 0 ]; then
    STEPS=("${ALL_STEPS[@]}")
else
    STEPS=("${FILTER_STEPS[@]}")
fi
NUM_STEPS=${#STEPS[@]}

# Initialize accumulators
step_cost_micro=()
step_input_tokens=()
step_output_tokens=()
for i in $(seq 0 $((NUM_STEPS - 1))); do
    step_cost_micro[$i]=0
    step_input_tokens[$i]=0
    step_output_tokens[$i]=0
done

total_cost_micro=0
total_input=0
total_output=0
benchmarks_seen=0
benchmarks_missing=0
report_lines=()

for dir in "$RESULTS_ROOT"/*/; do
    [ -d "$dir" ] || continue
    cost_file="${dir}cost.json"
    benchmark=$(basename "$dir")

    if [ ! -f "$cost_file" ]; then
        echo "[warn] no cost.json in $benchmark — skipping" >&2
        benchmarks_missing=$((benchmarks_missing + 1))
        continue
    fi

    benchmarks_seen=$((benchmarks_seen + 1))
    bench_total_micro=0

    for i in $(seq 0 $((NUM_STEPS - 1))); do
        step="${STEPS[$i]}"

        cost=$(jq -r --arg s "$step" '.[$s].cost // 0' "$cost_file")
        in_tok=$(jq -r --arg s "$step" '.[$s].input_tokens // 0' "$cost_file")
        out_tok=$(jq -r --arg s "$step" '.[$s].output_tokens // 0' "$cost_file")

        micro=$(echo "$cost" | awk '{printf "%d", $1 * 1000000}')
        in_int=$(echo "$in_tok" | awk '{printf "%d", $1}')
        out_int=$(echo "$out_tok" | awk '{printf "%d", $1}')

        step_cost_micro[$i]=$(( step_cost_micro[$i] + micro ))
        step_input_tokens[$i]=$(( step_input_tokens[$i] + in_int ))
        step_output_tokens[$i]=$(( step_output_tokens[$i] + out_int ))

        bench_total_micro=$(( bench_total_micro + micro ))
        total_input=$(( total_input + in_int ))
        total_output=$(( total_output + out_int ))
    done

    total_cost_micro=$(( total_cost_micro + bench_total_micro ))
    bench_total=$(echo "$bench_total_micro" | awk '{printf "%.4f", $1 / 1000000}')
    report_lines+=( "  $(printf '%-60s' "$benchmark")  \$$bench_total" )
done

# ── Print report ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  COST REPORT  —  $RESULTS_ROOT"
if [ ${#FILTER_STEPS[@]} -gt 0 ]; then
    echo "  Steps filtered: ${STEPS[*]}"
fi
echo "================================================================"

echo ""
echo "PER BENCHMARK"
echo "----------------------------------------------------------------"
for line in "${report_lines[@]}"; do
    echo "$line"
done

echo ""
echo "PER STEP"
echo "----------------------------------------------------------------"
for i in $(seq 0 $((NUM_STEPS - 1))); do
    step="${STEPS[$i]}"
    cost=$(echo "${step_cost_micro[$i]}" | awk '{printf "%.4f", $1 / 1000000}')
    in_k=$(echo "${step_input_tokens[$i]}" | awk '{printf "%'"'"'d", $1}')
    out_k=$(echo "${step_output_tokens[$i]}" | awk '{printf "%'"'"'d", $1}')
    printf "  %-45s  \$%s   (%s in / %s out tokens)\n" "$step" "$cost" "$in_k" "$out_k"
done

echo ""
echo "================================================================"
total=$(echo "$total_cost_micro" | awk '{printf "%.4f", $1 / 1000000}')
in_fmt=$(echo "$total_input"  | awk '{printf "%'"'"'d", $1}')
out_fmt=$(echo "$total_output" | awk '{printf "%'"'"'d", $1}')
echo "  TOTAL COST     : \$$total"
echo "  TOTAL TOKENS   : $in_fmt in / $out_fmt out"
echo "  BENCHMARKS     : $benchmarks_seen processed, $benchmarks_missing skipped"
echo "================================================================"
echo ""