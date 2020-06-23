#!/bin/bash

for r in {1..8}
do
	# Train model
	python3 ./trainer.py --run $r --model_name layer_norm_recurrent --train_steps 500000 --save_checkpoint_steps 100 200 500 1000 2000 10000 50000 100000 500000
	# Evaluate model on all test regions
	for t in {0..5}
	do
		python3 ./eval.py --run $r --test_set_ind $t --model_name layer_norm_recurrent --checkpoint 500000
	done
done
