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
		model_name = 'layer_norm'

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
		A_latent_mean, A_latent_var = tf.nn.moments(A_latent, 1)
		A_latent_SD = tf.sqrt(A_latent_var + eps)
		B_latent_mean, B_latent_var = tf.nn.moments(B_latent, 1)
		B_latent_SD = tf.sqrt(B_latent_var + eps)
		C_latent_mean, C_latent_var = tf.nn.moments(C_latent, 1)
		C_latent_SD = tf.sqrt(C_latent_var + eps)
		D_latent_mean, D_latent_var = tf.nn.moments(D_latent, 1)
		D_latent_SD = tf.sqrt(D_latent_var + eps)
		# Scale and shift parameters
		with tf.variable_scope('norm_params', reuse=tf.AUTO_REUSE) as scope:
			scale = tf.get_variable('scale', N_latent, initializer=tf.ones_initializer())
			shift = tf.get_variable('shift', N_latent, initializer=tf.zeros_initializer())
		# Normalize
		A_layer_norm = (((A_latent - tf.expand_dims(A_latent_mean,1)) / tf.expand_dims(A_latent_SD,1)) * scale) + shift
		B_layer_norm = (((B_latent - tf.expand_dims(B_latent_mean,1)) / tf.expand_dims(B_latent_SD,1)) * scale) + shift
		C_layer_norm = (((C_latent - tf.expand_dims(C_latent_mean,1)) / tf.expand_dims(C_latent_SD,1)) * scale) + shift
		D_layer_norm = (((D_latent - tf.expand_dims(D_latent_mean,1)) / tf.expand_dims(D_latent_SD,1)) * scale) + shift

		# [A, B, C, D] -> LSTM
		log.info('[A,B,C,D] -> LSTM...')
		D_score = scoring_model(A_layer_norm, B_layer_norm, C_layer_norm, D_layer_norm)

		# [A, B, C, foils] -> LSTM
		log.info('[A,B,C,foils] -> LSTM...')
		all_foil_score = []
		for foil in range(N_foils):
			# Extract latent rep for this foil
			this_foil_latent = all_foil_latent[:,foil,:]
			# Normalization
			# Normalization parameters
			foil_latent_mean, foil_latent_var = tf.nn.moments(this_foil_latent, 1)
			foil_latent_SD = tf.sqrt(foil_latent_var + eps)
			# Normalize
			foil_layer_norm = (((this_foil_latent - tf.expand_dims(foil_latent_mean,1)) / tf.expand_dims(foil_latent_SD,1)) * scale) + shift
			# Get score
			foil_score = scoring_model(A_layer_norm, B_layer_norm, C_layer_norm, foil_layer_norm)
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


