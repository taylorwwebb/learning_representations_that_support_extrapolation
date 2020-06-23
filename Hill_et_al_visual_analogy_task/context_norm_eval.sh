#!/bin/bash

# Train models
for r in {1..5}
do 
	python3 ./RPM_extrapolation_eval.py --norm_type context_norm --run $r
done