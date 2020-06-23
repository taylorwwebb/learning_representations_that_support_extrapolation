#!/bin/bash

for r in {1..8}
do
	# Train model
	python3 ./trainer.py --run $r --model_name batch_norm_dropout
	# Evaluate model on all test regions
	for t in {0..5}
	do
		python3 ./eval.py --run $r --test_set_ind $t --model_name batch_norm_dropout
	done
done
