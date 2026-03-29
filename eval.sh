#!/usr/bin/env python3

set -e 

python 1-test-scenario-construction.py --num-batch 2 
# --num-batch 3-5; we suggest having 3-5 batches to prevent hitting rate limits 
# in config.json make sure to specify ~10 configurtions 

python 2-simulation.py --b emotional-dependency --semaphore 3
# semaphore: supports 5 concurrent coroutines 
# --b the name of the benchmark folder 

python 3-evaluation.py --b emotional-dependency 

