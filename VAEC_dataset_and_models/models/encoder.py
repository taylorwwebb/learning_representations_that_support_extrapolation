# Import modules
import tensorflow as tf
import numpy as np
import os

# Import methods
from ops import *
from util import log

def extract_objects(imgs, ind):

	extracted_img = []
	batch_size = int(imgs.shape[0])
	for b in range(batch_size):
		extracted_img.append(tf.gather(imgs[b,:,:,:,:], ind[b], axis=0))
	extracted_img = tf.stack(extracted_img, axis=0)

	return extracted_img

def encoder(img):

	# Hyperparameters
	# Convolutional encoder
	encoder_N_conv_layers = 4
	encoder_N_conv_feature_maps = 32
	encoder_conv_stride = 2
	encoder_conv_kernel_size = 4
	encoder_conv_kernel_sizes = (np.ones(encoder_N_conv_layers) * encoder_conv_kernel_size).astype(np.int)
	encoder_conv_stride_sizes = (np.ones(encoder_N_conv_layers) * encoder_conv_stride).astype(np.int)
	encoder_conv_N_channels = np.ones(encoder_N_conv_layers).astype(np.int) * encoder_N_conv_feature_maps
	# FC encoder
	encoder_N_FC_layers = 2
	encoder_N_units_per_FC_layer = 256
	encoder_FC_size = np.ones((encoder_N_FC_layers), dtype = np.int) * encoder_N_units_per_FC_layer
	# Latent space
	N_latent = 256

	# Rescale image
	log.info('Scale image between 0 and 1...')
	img_scaled = img / 255.0

	# Encoder
	log.info('Encoder...')
	log.info('Convolutional layers...')
	encoder_conv_out = conv2d(img_scaled, encoder_conv_kernel_sizes, encoder_conv_N_channels, encoder_conv_stride_sizes, 
		scope='encoder_conv', reuse=tf.AUTO_REUSE)
	encoder_conv_out_shape = encoder_conv_out.shape
	log.info('FC layers...')
	encoder_conv_out_flat = tf.layers.flatten(encoder_conv_out)
	encoder_conv_out_flat_shape = int(encoder_conv_out_flat.shape[1])
	encoder_FC_out, encoder_FC_w, encoder_FC_biases = mlp(encoder_conv_out_flat, encoder_FC_size, scope='encoder_FC', reuse=tf.AUTO_REUSE)

	# Latent representation
	log.info('Latent representation...')
	latent, latent_linear_w, latent_linear_biases = linear_layer(encoder_FC_out, N_latent, scope="latent", reuse=tf.AUTO_REUSE)

	return latent

def encode_analogy_objs(imgs, ABCD, not_D):

	# Extract images for analogy terms (A, B, C, D) and foils (all objects != D)
	log.info('Extracting analogy terms...')
	A_img = extract_objects(imgs, ABCD[:, 0])
	B_img = extract_objects(imgs, ABCD[:, 1])
	C_img = extract_objects(imgs, ABCD[:, 2])
	D_img = extract_objects(imgs, ABCD[:, 3])
	not_D_imgs = extract_objects(imgs, not_D)

	# Get latent codes
	log.info('Building encoders...')
	log.info('A...')
	A_latent = encoder(A_img)
	log.info('B...')
	B_latent = encoder(B_img)
	log.info('C...')
	C_latent = encoder(C_img)
	log.info('D...')
	D_latent = encoder(D_img)
	log.info('Foils...')
	all_foil_latent = []
	N_foils = int(not_D.shape[1])
	for foil in range(N_foils):
		log.info('foil ' + str(foil+1) + '...')
		all_foil_latent.append(encoder(tf.gather(not_D_imgs, foil, axis=1)))
	all_foil_latent = tf.stack(all_foil_latent, axis=1)

	return A_latent, B_latent, C_latent, D_latent, all_foil_latent