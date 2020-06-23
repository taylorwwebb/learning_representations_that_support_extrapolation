#!/bin/bash

# Train models
for r in {1..5}
do 
	python3 ./RPM_extrapolation_eval.py --norm_type source_targ_norm --run $r
done