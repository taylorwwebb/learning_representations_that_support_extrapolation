#!/bin/bash

for r in {1..8}
do
	# Train model
	python3 ./trainer.py --run $r --model_name batch_norm_train_stats
	# Get training set stats
	python3 ./norm_stats.py --run $r
	# Evaluate model on all test regions
	for t in {0..5}
	do
		python3 ./eval_train_stats.py --run $r --test_set_ind $t
	done
done
