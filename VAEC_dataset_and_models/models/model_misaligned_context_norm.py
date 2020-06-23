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
		model_name = 'misaligned_context_norm'

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
		# Scale and shift parameters
		with tf.variable_scope('norm_params', reuse=tf.AUTO_REUSE) as scope:
			scale = tf.get_variable('scale', N_latent, initializer=tf.ones_initializer())
			shift = tf.get_variable('shift', N_latent, initializer=tf.zeros_initializer())
		# Concate objects along temporal dimension / flatten along batch dimension
		ABCD_flat = []
		for b in range(batch_size):
			ABCD_flat.append(tf.stack([A_latent[b,:], B_latent[b,:], C_latent[b,:], D_latent[b,:]], axis=0))
		ABCD_flat = tf.concat(ABCD_flat, axis=0)
		# Normalize over smaller sub-batches of 5
		N_sub_batches = int(np.round(((batch_size * 4) / 5)))
		all_sub_batch_norm = []
		for b in range(N_sub_batches):
			# Extract sub-batch
			if ((b+1) * 5) > (batch_size * 4):
				sub_batch_ind = np.arange(b * 5, (batch_size * 4))
			else:
				sub_batch_ind = np.arange(b * 5, (b+1) * 5)
			sub_batch = tf.gather(ABCD_flat, sub_batch_ind, axis=0)
			# Normalization parameters
			sub_batch_latent_mean, sub_batch_latent_var = tf.nn.moments(sub_batch, 0)
			sub_batch_latent_SD = tf.sqrt(sub_batch_latent_var + eps)
			# Normalize
			sub_batch_norm = (((sub_batch - sub_batch_latent_mean) / sub_batch_latent_SD) * scale) + shift
			# Add to list
			all_sub_batch_norm.append(sub_batch_norm)
		# Concatenate
		all_sub_batch_norm = tf.concat(all_sub_batch_norm, axis=0)
		# Separate back out into individual objects
		ABCD_sub_batch_norm = []
		for b in range(batch_size):
			ABCD_ind = np.arange(b * 4, (b+1) * 4)
			ABCD_sub_batch_norm.append(tf.gather(all_sub_batch_norm, ABCD_ind, axis=0))
		ABCD_sub_batch_norm = tf.stack(ABCD_sub_batch_norm, axis=1)
		A_batch_norm = ABCD_sub_batch_norm[0,:,:]
		B_batch_norm = ABCD_sub_batch_norm[1,:,:]
		C_batch_norm = ABCD_sub_batch_norm[2,:,:]
		D_batch_norm = ABCD_sub_batch_norm[3,:,:]

		# [A, B, C, D] -> LSTM
		log.info('[A,B,C,D] -> LSTM...')
		D_score = scoring_model(A_batch_norm, B_batch_norm, C_batch_norm, D_batch_norm)

		# [A, B, C, foils] -> LSTM
		log.info('[A,B,C,foils] -> LSTM...')
		all_foil_score = []
		for foil in range(N_foils):
			# Extract latent rep for this foil
			this_foil_latent = all_foil_latent[:,foil,:]
			# Concate objects along temporal dimension / flatten along batch dimension
			ABCfoil_flat = []
			for b in range(batch_size):
				ABCfoil_flat.append(tf.stack([A_latent[b,:], B_latent[b,:], C_latent[b,:], this_foil_latent[b,:]], axis=0))
			ABCfoil_flat = tf.concat(ABCfoil_flat, axis=0)
			# Normalize over smaller sub-batches of 5
			N_sub_batches = int(np.round(((batch_size * 4) / 5)))
			all_sub_batch_norm = []
			for b in range(N_sub_batches):
				# Extract sub-batch
				if ((b+1) * 5) > (batch_size * 4):
					sub_batch_ind = np.arange(b * 5, (batch_size * 4))
				else:
					sub_batch_ind = np.arange(b * 5, (b+1) * 5)
				sub_batch = tf.gather(ABCfoil_flat, sub_batch_ind, axis=0)
				# Normalization parameters
				sub_batch_latent_mean, sub_batch_latent_var = tf.nn.moments(sub_batch, 0)
				sub_batch_latent_SD = tf.sqrt(sub_batch_latent_var + eps)
				# Normalize
				sub_batch_norm = (((sub_batch - sub_batch_latent_mean) / sub_batch_latent_SD) * scale) + shift
				# Add to list
				all_sub_batch_norm.append(sub_batch_norm)
			# Concatenate
			all_sub_batch_norm = tf.concat(all_sub_batch_norm, axis=0)
			# Separate back out into individual objects
			ABCfoil_sub_batch_norm = []
			for b in range(batch_size):
				ABCfoil_ind = np.arange(b * 4, (b+1) * 4)
				ABCfoil_sub_batch_norm.append(tf.gather(all_sub_batch_norm, ABCfoil_ind, axis=0))
			ABCfoil_sub_batch_norm = tf.stack(ABCfoil_sub_batch_norm, axis=1)
			A_batch_norm = ABCfoil_sub_batch_norm[0,:,:]
			B_batch_norm = ABCfoil_sub_batch_norm[1,:,:]
			C_batch_norm = ABCfoil_sub_batch_norm[2,:,:]
			foil_batch_norm = ABCfoil_sub_batch_norm[3,:,:]
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


