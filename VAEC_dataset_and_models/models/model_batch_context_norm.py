import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

# Import methods
from ops import *
from encoder import *
from analogy_scoring_model import *
from util import log

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

# Method for creating input variable names
def create_input_var_names():
	input_var_names = [
					'imgs',
					'ABCD',
					'not_D',]
	return input_var_names

class Model(object):

	def __init__(self, config, batch_ops, is_train=True):

		# Model name
		model_name = 'batch_context_norm'

		# Model inputs
		imgs = batch_ops['imgs']
		ABCD = tf.cast(batch_ops['ABCD'], dtype=tf.int64)
		not_D = tf.cast(batch_ops['not_D'], dtype=tf.int64)

		# Dimensions
		batch_size = int(config.batch_size)
		N_foils = int(not_D.shape[1])

		# Get latent codes for all images
		A_latent, B_latent, C_latent, D_latent, all_foil_latent = encode_analogy_objs(imgs, ABCD, not_D)
		N_latent = int(A_latent.shape[1])

		# Normalization
		# Small constant (for avoiding division by zero)
		eps = 1e-8
		# Flatten batch along sequence dimension
		flat_seq = []
		for b in range(batch_size):
			flat_seq.append(A_latent[b, :])
			flat_seq.append(B_latent[b, :])
			flat_seq.append(C_latent[b, :])
			flat_seq.append(D_latent[b, :])
		flat_seq = tf.stack(flat_seq, axis=0)
		# Normalization parameters 
		ABCD_mean, ABCD_var = tf.nn.moments(flat_seq, 0)
		ABCD_SD = tf.sqrt(ABCD_var + eps)
		# Scale and shift parameters
		with tf.variable_scope('norm_params', reuse=tf.AUTO_REUSE) as scope:
			scale = tf.get_variable('scale', N_latent, initializer=tf.ones_initializer())
			shift = tf.get_variable('shift', N_latent, initializer=tf.zeros_initializer())
		# Normalize
		A_batch_norm = (((A_latent - ABCD_mean) / ABCD_SD) * scale) + shift
		B_batch_norm = (((B_latent - ABCD_mean) / ABCD_SD) * scale) + shift
		C_batch_norm = (((C_latent - ABCD_mean) / ABCD_SD) * scale) + shift
		D_batch_norm = (((D_latent - ABCD_mean) / ABCD_SD) * scale) + shift

		# [A, B, C, D] -> LSTM
		log.info('[A,B,C,D] -> LSTM...')
		D_score = scoring_model(A_batch_norm, B_batch_norm, C_batch_norm, D_batch_norm)

		# [A, B, C, foils] -> LSTM
		log.info('[A,B,C,foils] -> LSTM...')
		all_foil_score = []
		for foil in range(N_foils):
			# Extract latent rep for this foil
			this_foil_latent = all_foil_latent[:,foil,:]
			# Normalization
			# Flatten batch along sequence dimension
			flat_seq = []
			for b in range(batch_size):
				flat_seq.append(A_latent[b, :])
				flat_seq.append(B_latent[b, :])
				flat_seq.append(C_latent[b, :])
				flat_seq.append(this_foil_latent[b, :])
			flat_seq = tf.stack(flat_seq, axis=0)
			# Normalization parameters 
			ABCfoil_mean, ABCfoil_var = tf.nn.moments(flat_seq, 0)
			ABCfoil_SD = tf.sqrt(ABCfoil_var + eps)
			# Normalize
			A_batch_norm = (((A_latent - ABCfoil_mean) / ABCfoil_SD) * scale) + shift
			B_batch_norm = (((B_latent - ABCfoil_mean) / ABCfoil_SD) * scale) + shift
			C_batch_norm = (((C_latent - ABCfoil_mean) / ABCfoil_SD) * scale) + shift
			foil_batch_norm = (((this_foil_latent - ABCfoil_mean) / ABCfoil_SD) * scale) + shift
			# Get score
			foil_score = scoring_model(A_batch_norm, B_batch_norm, C_batch_norm, foil_batch_norm)
			# Accumulate foil scores
			all_foil_score.append(foil_score)

		# Concatenate all scores
		all_foil_score = tf.concat(all_foil_score, axis=1)
		all_scores = tf.concat([D_score, all_foil_score], axis=1)
		all_scores_softmax = tf.nn.softmax(all_scores)

		# Loss
		log.info("Loss (cross-entropy over candidate scores)...")
		targets = tf.concat([tf.ones(D_score.shape), tf.zeros(all_foil_score.shape)], axis=1)
		self.train_loss, accuracy, correct_preds = build_cross_entropy_loss(all_scores, targets)
		accuracy = accuracy * 100.0

		# Model outputs
		self.all_out = {
						'accuracy': accuracy}
