# Learning Representations that Support Extrapolation

Code for the paper **Learning Representations that Support Extrapolation**.

## Visual Analogy Extrapolation Challenge (VAEC)

The directory `./VAEC_dataset_and_models` contains code for generating the VAEC dataset, and code for all models used for this dataset. To generate the VAEC dataset, navigate to `./VAEC_dataset_and_models/dset_gen/` and run the following command to generate the Translation Extrapolation Regime:
```
python3 ./VAEC_trans_extrap.py
```
and the following command to generate the Scale Extrapolation Regime:
```
python3 ./VAEC_scale_extrap.py
```

To train and evaluate any of the models described in the paper, simply run the corresponding the script in the `./VAEC_dataset_and_models/scripts/` directory. For instance, to train and evaluate networks with context normalization on the Translation Extrapolation Regime, navigate to `./VAEC_dataset_and_models/`, and run the following command:
```
./scripts/context_norm_trans_extrap.sh
```
Similarly, to train and evaluate networks with context normalization on the Scale Extrapolation Regime, run the following command:
```
./scripts/context_norm_scale_extrap.sh
```

## Dynamic Object Prediction

The directory `./dynamic_object_prediction` contains code to generate the dynamic object prediction task, and code for all models used on this task.

## Visual Analogy Task from *Learning to Make Analogies by Contrasting Abstract Relational Structure*

The directory `./LABC_visual_analogy_task` contains code for all models used on the dataset from [Learning to Make Analogies by Contrasting Abstract Relational Structure](https://arxiv.org/abs/1902.00120).

The dataset can be downloaded from https://github.com/deepmind/abstract-reasoning-matrices. Download the extrapolation dataset (inside the `./analogies` directory). Create a directory inside `./LABC_visual_analogy_task/datasets/` called `extrapolation`, and place the files for the dataset in this directory.

To train and evaluate any of the models described in the paper, run the corresponding script. For instance, to train networks using source/target context normalization, navigate to `./LABC_visual_analogy_task/` and run the following command:
```
./source_targ_norm_train.sh
```
To evaluate these networks, run the following sommand:
```
./source_targ_norm_eval.sh
```

## Prerequisites

- Python 3
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [colorlog](https://github.com/borntyping/python-colorlog)
For VAEC dataset:
- [Tensorflow 1.13.1](https://www.tensorflow.org/)
- [h5py](http://docs.h5py.org/en/latest/)
- [PIL](https://pillow.readthedocs.io/en/3.1.x/installation.html)
For Dynamic Object Prediction Task and Visual Analogy Task from *Learning to Make Analogies by Contrasting Abstract Relational Structure*:
- [PyTorch](https://pytorch.org/)
