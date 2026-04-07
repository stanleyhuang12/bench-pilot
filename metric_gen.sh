#!/usr/bin/env bash

set -e 
set -o pipefail

results_root=results
output_file=descriptions.md
trap 'echo "[ERROR] Script failed at line $LINENO — command: $BASH_COMMAND" >&2' ERR

> "$output_file"
# echo "Generating metrics and scenario descriptions for each benchmark."

# echo "Doing a dry run just to validate procedure" 

# Run dry runs 
# python 0-parse-through-xlsx.py --csv benchmark_submission.csv --results-root $results_root --dry-run


# Loop through benchmark_submissions.csv and perform goal normalization and metric definition 
# for i in {1..25}; do 
#     echo "Benchmark: $i"
#     python 0-parse-through-xlsx.py --csv benchmark_submission.csv --results-root $results_root --row-entry "$i"
#     # we are looping past index 0 because the first index is emotional-dependency and we do not want to overwrite that goal.json file 
# done 

# For every dir under the root result directory, we construct scenarios and metrics 
skip_benchmark="emotional-dependency" 

echo "Looping through benchmark scenario construction" 

# for dir in "$results_root"/*; do
#     [ -d "$dir" ] || continue
#     benchmark_name=$(basename "$dir")
#     [ "$benchmark_name" = "$skip_benchmark" ] && continue 
#     python3 1-test-scenario-construction.py \
#         --b "$benchmark_name" \
#         --results-root "$results_root" \
#         --overspecification \
#         --demographics gender age \
#         --overwrite 
# done
# wait

NUM_SCENE_EXTRACT=3 # number of random scenarios to extract
 
format_cost() {
    local val="$1" label="$2"
    if [ "$val" = "null" ] || [ -z "$val" ]; then
        echo "$label: N/A" >> "$output_file"
    else
        echo "$label: \$$(printf '%.4f' "$val")" >> "$output_file"
    fi
}
 
for file in "$results_root"/*; do
    [ -d "$file" ] || continue
    echo "Retrieving metrics and data  from $file"
    goal_file="${file}/goal.json"
    test_file="${file}/test.json"
    cost_file="${file}/cost.json"
 
    [ -f "$goal_file" ] || { echo "Warning: missing $goal_file — skipping" >&2; continue; }
    [ -f "$test_file" ] || { echo "Warning: missing $test_file — skipping" >&2; continue; }
    [ -f "$cost_file" ] || { echo "Warning: missing $cost_file — skipping" >&2; continue; }
 
    echo "# $goal_file" >> "$output_file"
    
    echo "### Metrics" >>"$output_file"
    while read -r metric; do
        id=$(echo "$metric" | jq -r '.id')
        name=$(echo "$metric" | jq -r '.metric_name')
        definition=$(echo "$metric" | jq -r '.definition')
        echo "ID: $id | Metric Name: $name" >> "$output_file"
        echo "Definition: $definition" >> "$output_file"
        echo >> "$output_file"
    done < <(jq -c ".metric[]" "$goal_file")
    
    echo "### Scenarios" >> "$output_file"
    while read -r scenario; do
        scenario_id=$(echo "$scenario" | jq -r '.id')
        scenario_title=$(echo "$scenario" | jq -r '.title')
        scenario_description=$(echo "$scenario" | jq -r '.description')
        echo "Scenario ID: $scenario_id | Scenario Title: $scenario_title" >> "$output_file"
        echo "Scenario Description: $scenario_description" >> "$output_file"
        echo >> "$output_file"
    done < <(jq -c ".scenarios[]" "$test_file" | sort -R | head -n "$NUM_SCENE_EXTRACT")
 
    echo >> "$output_file"
    echo "### Costs" >> "$output_file"
 
    goal_cost=$(jq -r '.goal_normalization.cost                    // "null"' "$cost_file")
    base_cost=$(jq -r '.base_scenario_construction.cost            // "null"' "$cost_file")
    demographic_cost=$( jq -r '.demographic_scenario_construction.cost     // "null"' "$cost_file")
 
    format_cost "$goal_cost" "Goal Normalization"
    format_cost "$base_cost" "Base Scenario Construction"
    format_cost "$demographic_cost" "Demographic Scenario Construction"
 
    printf '–%.0s' {1..64} >> "$output_file"
    echo >> "$output_file"
    echo >> "$output_file"
 
done
 