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
		model_name = 'sliding_context_norm'

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

		# Sliding-window context normalization
		window_size = 4
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
		# Zero-pad beginning of sequence with end of sequence
		flat_seq_size = int(flat_seq.shape[0])
		end_of_seq = flat_seq[flat_seq_size-window_size:,:]
		flat_seq = tf.concat([end_of_seq, flat_seq], axis=0)
		# Compute normalization parameters separately for each time-point
		sliding_mean = []
		sliding_SD = []
		for t in range((window_size),flat_seq_size + window_size):
			# Indices
			t_start = t - window_size
			# Normalization parameters 
			mn, var = tf.nn.moments(flat_seq[t_start:t, :], 0)
			SD = tf.sqrt(var + eps)
			sliding_mean.append(mn)
			sliding_SD.append(SD)
		sliding_mean = tf.stack(sliding_mean, axis=0)
		sliding_SD = tf.stack(sliding_SD, axis=0)
		# Scale and shift parameters
		with tf.variable_scope('norm_params', reuse=tf.AUTO_REUSE) as scope:
			scale = tf.get_variable('scale', N_latent, initializer=tf.ones_initializer())
			shift = tf.get_variable('shift', N_latent, initializer=tf.zeros_initializer())
		# Remove zero-padding
		flat_seq = flat_seq[window_size:,:]
		# Normalize entire sequence
		flat_seq_norm = (((flat_seq - sliding_mean) / sliding_SD) * scale) + shift
		# Separate out analogy terms
		A_ind = np.arange(batch_size) * 4
		A_batch_norm = tf.gather(flat_seq_norm, A_ind)
		B_ind = (np.arange(batch_size) * 4) + 1
		B_batch_norm = tf.gather(flat_seq_norm, B_ind)
		C_ind = (np.arange(batch_size) * 4) + 2
		C_batch_norm = tf.gather(flat_seq_norm, C_ind)
		D_ind = (np.arange(batch_size) * 4) + 3
		D_batch_norm = tf.gather(flat_seq_norm, D_ind)

		# [A, B, C, D] -> LSTM
		log.info('[A,B,C,D] -> LSTM...')
		D_score = scoring_model(A_batch_norm, B_batch_norm, C_batch_norm, D_batch_norm)

		# [A, B, C, foils] -> LSTM
		log.info('[A,B,C,foils] -> LSTM...')
		all_foil_score = []
		for foil in range(N_foils):
			# Extract latent rep for this foil
			this_foil_latent = all_foil_latent[:,foil,:]
			# Batch-normalization
			# Flatten batch along sequence dimension
			flat_seq = []
			for b in range(batch_size):
				flat_seq.append(A_latent[b, :])
				flat_seq.append(B_latent[b, :])
				flat_seq.append(C_latent[b, :])
				flat_seq.append(this_foil_latent[b, :])
			flat_seq = tf.stack(flat_seq, axis=0)
			# Zero-pad beginning of sequence with end of sequence
			flat_seq_size = int(flat_seq.shape[0])
			end_of_seq = flat_seq[flat_seq_size-window_size:,:]
			flat_seq = tf.concat([end_of_seq, flat_seq], axis=0)
			# Compute normalization parameters separately for each time-point
			sliding_mean = []
			sliding_SD = []
			for t in range((window_size),flat_seq_size + window_size):
				# Indices
				t_start = t - window_size
				# Normalization parameters 
				mn, var = tf.nn.moments(flat_seq[t_start:t, :], 0)
				SD = tf.sqrt(var + eps)
				sliding_mean.append(mn)
				sliding_SD.append(SD)
			sliding_mean = tf.stack(sliding_mean, axis=0)
			sliding_SD = tf.stack(sliding_SD, axis=0)
			# Remove zero-padding
			flat_seq = flat_seq[window_size:,:]
			# Normalize entire sequence
			flat_seq_norm = (((flat_seq - sliding_mean) / sliding_SD) * scale) + shift
			# Separate out analogy terms
			A_ind = np.arange(batch_size) * 4
			A_batch_norm = tf.gather(flat_seq_norm, A_ind)
			B_ind = (np.arange(batch_size) * 4) + 1
			B_batch_norm = tf.gather(flat_seq_norm, B_ind)
			C_ind = (np.arange(batch_size) * 4) + 2
			C_batch_norm = tf.gather(flat_seq_norm, C_ind)
			foil_ind = (np.arange(batch_size) * 4) + 3
			foil_batch_norm = tf.gather(flat_seq_norm, foil_ind)
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
