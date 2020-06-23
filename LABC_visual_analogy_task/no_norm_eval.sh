#!/bin/bash

# Train models
for r in {1..5}
do 
	python3 ./RPM_extrapolation_eval.py --run $r
done