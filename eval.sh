#!/usr/bin/env python3

set -e 

python 1-test-scenario-construction.py --num-batch 2 

python 2-simulation.py --b emotional-dependency --semaphore 3

python 3-evaluation.py --b emotional-dependency 