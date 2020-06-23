#!/bin/bash

# Train models
for r in {1..5}
do 
	python3 ./RPM_extrapolation.py --norm_type batch_norm --run $r
done