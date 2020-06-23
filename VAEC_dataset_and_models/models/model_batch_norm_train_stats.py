import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
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
		model_name = 'batch_norm_train_stats'

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
		# Normalization parameters
		if is_train:
			A_latent_mean, A_latent_var = tf.nn.moments(A_latent, 0)
			B_latent_mean, B_latent_var = tf.nn.moments(B_latent, 0)
			C_latent_mean, C_latent_var = tf.nn.moments(C_latent, 0)
			D_latent_mean, D_latent_var = tf.nn.moments(D_latent, 0)
		else:
			A_latent_mean = tf.constant(np.array(config.norm_stats['A_latent_mean']))
			A_latent_var = tf.constant(np.array(config.norm_stats['A_latent_var']))
			B_latent_mean = tf.constant(np.array(config.norm_stats['B_latent_mean']))
			B_latent_var = tf.constant(np.array(config.norm_stats['B_latent_var']))
			C_latent_mean = tf.constant(np.array(config.norm_stats['C_latent_mean']))
			C_latent_var = tf.constant(np.array(config.norm_stats['C_latent_var']))
			D_latent_mean = tf.constant(np.array(config.norm_stats['D_latent_mean']))
			D_latent_var = tf.constant(np.array(config.norm_stats['D_latent_var']))
		A_latent_SD = tf.sqrt(A_latent_var + eps)
		B_latent_SD = tf.sqrt(B_latent_var + eps)
		C_latent_SD = tf.sqrt(C_latent_var + eps)
		D_latent_SD = tf.sqrt(D_latent_var + eps)
		# Scale and shift parameters
		with tf.variable_scope('norm_params', reuse=tf.AUTO_REUSE) as scope:
			scale = tf.get_variable('scale', A_latent_mean.shape, initializer=tf.ones_initializer())
			shift = tf.get_variable('shift', A_latent_mean.shape, initializer=tf.zeros_initializer())
		# Normalize
		A_batch_norm = (((A_latent - A_latent_mean) / A_latent_SD) * scale) + shift
		B_batch_norm = (((B_latent - B_latent_mean) / B_latent_SD) * scale) + shift
		C_batch_norm = (((C_latent - C_latent_mean) / C_latent_SD) * scale) + shift
		D_batch_norm = (((D_latent - D_latent_mean) / D_latent_SD) * scale) + shift

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
			# Small constant (for avoiding division by zero)
			eps = 1e-8
			# Normalization parameters
			if is_train:
				foil_latent_mean, foil_latent_var = tf.nn.moments(this_foil_latent, 0)
			else:
				foil_latent_mean = tf.constant(np.array(config.norm_stats['all_foil_latent_mean'])[foil,:])
				foil_latent_var = tf.constant(np.array(config.norm_stats['all_foil_latent_var'])[foil,:])
			foil_latent_SD = tf.sqrt(foil_latent_var + eps)
			# Normalize
			foil_batch_norm = (((this_foil_latent - foil_latent_mean) / foil_latent_SD) * scale) + shift
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
		self.all_out_stats = {'A_latent': A_latent,
							  'B_latent': B_latent,
							  'C_latent': C_latent,
							  'D_latent': D_latent,
							  'all_foil_latent': all_foil_latent}

