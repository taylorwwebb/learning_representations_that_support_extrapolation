#!/bin/bash

for r in {1..8}
do
	# Train model
	python3 ./trainer.py --run $r --model_name no_norm --learning_rate 1e-4 --train_steps 500000 --save_checkpoint_steps 100 200 500 1000 2000 10000 50000 100000 500000 --dset_names scale_train scale_test1 scale_test2 scale_test3 scale_test4 scale_test5
	# Evaluate model on all test regions
	for t in {0..5}
	do
		python3 ./eval.py --run $r --test_set_ind $t --model_name no_norm --checkpoint 500000 --test_set_names scale_train scale_test1 scale_test2 scale_test3 scale_test4 scale_test5
	done
done
