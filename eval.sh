#!/usr/bin/env bash

set -e 

results_root=results
BENCHMARK=emotional-dependency # benchmark name 
num_batch=2 # number of test scenario construction batches 
ns=3 # resampling parameter | btw, this is the most time intensive parameter 
semaphore=8 # number of concurrent operations async can handle 

num_scenarios=$(jq '.generation.num_scenarios' config.json)
turns_per_conversations=$(jq '.generation.turns_per_conversation' config.json)

echo "
Running eval pipeline on $BENCHMARK. 

Generating $((num_scenarios * num_batch)) scenarios, ${num_scenarios} scenarios per call and ${num_batch} batch 
Generating $((num_scenarios * num_batch * ns)) simulations:
  - Scenario batches: $num_batch
  - Resamples per batch: $ns
  - Parallelism: $semaphore" >> time_logger.txt 


log_step() {
    local stepname=$1

    local start=$(date +%s)
    echo "${stepname}:  $(date '+%Y-%m-%d %H:%M:%S')" >> time_logger.txt 

    shift 
    "$@" #runs the commands after the first local var 

    local end=$(date +%s)
    echo "${stepname}:  $(date '+%Y-%m-%d %H:%M:%S')" >> time_logger.txt 

    elapsed_time=$((end - start))
    echo "${stepname}: ${elapsed_time}" >> time_logger.txt 
}


total_start=$(date +%s)

log_step "Construct scenarios" python 1-test-scenario-construction.py --num-batch ${num_batch} --benchmark ${BENCHMARK} 
# --num-batch 3-5; we suggest having 3-5 batches to prevent hitting rate limits 
# in config.json make sure to specify ~10 configurtions 


log_step "Generate simulations" python 2-simulation.py --b ${BENCHMARK} --semaphore ${semaphore} --ns ${ns}
# semaphore: supports upper bound of 5 concurrent coroutines automatically looping through all scenario simulations
    # note that 5 semaphore is not a strict condition.. we can switch to boundedsemaphore 
# --b the name of the benchmark folder 
# --ns the number of samples 

log_step "Evaluate" python 3-evaluation.py --b ${BENCHMARK}
log_step "Export" python 4-export.py --b ${BENCHMARK}

total_end=$(date +%s)
# run with bash eval.sh 

total_run_time=$((total_end - total_start))
echo "Total pipeline runtime: $total_run_time seconds" >> time_logger.txt


