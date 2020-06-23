# Import modules
import tensorflow as tf
import numpy as np
import os

# Import methods
from ops import *
from util import log

def scoring_model(A, B, C, D, layernorm=False):

	LSTM_size = 256
	batch_size = int(A.shape[0])
	# Add tags to each element
	A_tag = tf.tile(tf.expand_dims(tf.constant([1, 0, 0, 0], dtype=tf.float32), 0), [batch_size, 1])
	A_tagged = tf.concat([A_tag, A], axis=1)
	B_tag = tf.tile(tf.expand_dims(tf.constant([0, 1, 0, 0], dtype=tf.float32), 0), [batch_size, 1])
	B_tagged = tf.concat([B_tag, B], axis=1)
	C_tag = tf.tile(tf.expand_dims(tf.constant([0, 0, 1, 0], dtype=tf.float32), 0), [batch_size, 1])
	C_tagged = tf.concat([C_tag, C], axis=1)
	D_tag = tf.tile(tf.expand_dims(tf.constant([0, 0, 0, 1], dtype=tf.float32), 0), [batch_size, 1])
	D_tagged = tf.concat([D_tag, D], axis=1)
	# Concatenate inputs as timeseries
	ABCD_lstm_inputs = tf.stack([A_tagged, B_tagged, C_tagged, D_tagged], axis=0)
	# LSTM
	log.info("LSTM...")
	if layernorm:
		lstm_all_output = lstm_layernorm(ABCD_lstm_inputs, LSTM_size, scope="lstm", reuse=tf.AUTO_REUSE)
	else:
		lstm_all_output = lstm(ABCD_lstm_inputs, LSTM_size, scope="lstm", reuse=tf.AUTO_REUSE)
	lstm_final_output = lstm_all_output[-1, :, :]
	# Linear layer
	log.info("Linear scoring layer...")
	score, score_w, score_biases = linear_layer(lstm_final_output, 1, scope='score_linear', reuse=tf.AUTO_REUSE)

	return score