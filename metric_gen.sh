#!/usr/bin/env bash

set -e 
set -o pipefail

results_root=results
output_file=descriptions.txt

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

for dir in "$results_root"/*; do
    [ -d "$dir" ] || continue
    benchmark_name=$(basename "$dir")
    [ "$benchmark_name" = "$skip_benchmark" ] && continue 
    python3 1-test-scenario-construction.py \
        --b "$benchmark_name" \
        --results-root "$results_root" \
        --overspecification \
        --demographics gender age \
        --overwrite 
done
wait

NUM_SCENE_EXTRACT=3 # number of random scenarios to extract 
for file in "$results_root"/*; do
    [ -d "$file" ] || continue #filter for the file 
    echo "Retrieving metrics from $file " #log info 
    goal_file="${file}/goal.json" #locate the path 
    test_file="${file}/test.json"
    cost_file="${file}/cost.json"

    #Loop through each file 
    echo "$goal_file" >> $output_file
    
    #For each file, extract and document the metrics 
    jq -c ".metric[]" "$goal_file" | while read -r metric; do 
        id=$(echo "$metric" | jq -r '.id')
        name=$(echo "$metric" | jq -r '.metric_name')
        definition=$(echo "$metric" | jq -r '.definition')
        echo "ID: $id | Metric Name: $name" >> $output_file
        echo "Definition: $definition" >> $output_file
        echo >> $output_file
    done 

    #For each file, extract and document N=NUM_SCENE_EXTRACT of scenarios 
    jq -c ".scenarios[]" "${test_file}" | shuf -n "$NUM_SCENE_EXTRACT" | while read -r scenario; do 
        scenario_id=$(echo "$scenario" | jq -r '.id')
        scenario_title=$(echo "$scenario" | jq -r '.title')
        scenario_description=$(echo "$scenario" | jq -r '.description')

        echo "Scenario ID: $scenario_id | Scenario Title: $scenario_title" >> $output_file
        echo "Scenario Description: $scenario_description" >> $output_file
        echo >> $output_file
    done 

    #For each file, extract and document the costs it would take to run this 
    echo >> $output_file

    echo "Costs" >> "$output_file"
    goal_cost=$(jq -r '.goal_normalization.cost' "$cost_file")
    base_cost=$(jq -r '.base_scenario_construction.cost' "$cost_file")
    demographic_cost=$(jq -r '.demographic_scenario_construction.cost' "$cost_file")

    echo "Goal Normalization: \$$(printf '%.2f' "$goal_cost")" >> "$output_file"
    echo "Base Scenario Construction: \$$(printf '%.2f' "$base_cost")" >> "$output_file"
    echo "Demographic Scenario Construction: \$$(printf '%.2f' "$demographic_cost")" >> "$output_file"
    
    printf '–%.0s' {1..64} >> $output_file
    printf '–%.0s' {1..64} >> $output_file
    echo >> $output_file

done

