#!/bin/bash

for r in {1..8}
do
	# Train model
	python3 ./trainer.py --run $r --model_name batch_norm --dset_names scale_train scale_test1 scale_test2 scale_test3 scale_test4 scale_test5
	# Evaluate model on all test regions
	for t in {0..5}
	do
		python3 ./eval.py --run $r --test_set_ind $t --model_name batch_norm --test_set_names scale_train scale_test1 scale_test2 scale_test3 scale_test4 scale_test5
	done
done
