# Modules
import h5py
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import argparse
import sys
from itertools import product
from collections import OrderedDict

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

# Get log function from util (in parent directory)
sys.path.insert(0,'../')
from util import log

# Set pyplot to only write to file, not to render to window (necessary for using pyplot on cluster)
plt.switch_backend('agg')

def generator(config):

	def gen_image(img_size, bg_color, obj_color, size, x, y):

		# Generate image and drawer
		img = Image.new('RGB', (img_size, img_size), color=bg_color)
		drawer = ImageDraw.Draw(img)

		# Generate coordinates for corners
		left = x - (size/2)
		right = x + (size/2)
		top = y - (size/2)
		bottom = y + (size/2)
		position = [(left, top), 
					(right, bottom)]

		# Draw image
		drawer.rectangle(position, fill=obj_color)

		# Convert image to array
		img = np.array(img)

		return img

	def draw_images(imgs, ABCD, dir_name, a, dim):

		# A
		plt.subplot(2,2,1)
		plt.imshow(imgs[ABCD[0], :, :, :], interpolation='none')
		plt.grid()
		plt.title('A = ' + str(ABCD[0]))
		# B
		plt.subplot(2,2,2)
		plt.imshow(imgs[ABCD[1], :, :, :], interpolation='none')
		plt.grid()
		plt.title('B = ' + str(ABCD[1]))
		# C
		plt.subplot(2,2,3)
		plt.imshow(imgs[ABCD[2], :, :, :], interpolation='none')
		plt.grid()
		plt.title('C = ' + str(ABCD[2]))
		# D
		plt.subplot(2,2,4)
		plt.imshow(imgs[ABCD[3], :, :, :], interpolation='none')
		plt.grid()
		plt.title('D = ' + str(ABCD[3]))
		# Filename
		dim_names = ['X', 'Y', 'size', 'brightness']
		fname = dir_name + str(a) + '_' + dim_names[dim] + '.png'
		plt.savefig(fname)
		# Close plot
		plt.close()

	#############################################################################################################################
	############################# Generate all analogies possible within each train/test region #################################
	#############################################################################################################################

	log.info('Generating all analogies possible (within a given train/test region)...')

	# All unidimensional pairs 
	all_AB = []
	for A in range(config.N_levels_per_region):
		for B in range(config.N_levels_per_region):
			AB_dist = B - A
			# Exclude same-object pairs
			same_object = AB_dist == 0
			# Exclude largest possible distance (because this precludes analogy)
			largest_dist = np.abs(AB_dist) == (config.N_levels_per_region - 1)
			# Add to list
			if not same_object and not largest_dist:
				all_AB.append([A, B, AB_dist])
	# Convert to array
	all_AB = np.array(all_AB)

	# Sort pairs by distance
	all_dist = all_AB[:,2]
	unique_dist = np.unique(all_dist)
	all_AB_dist_sorted = OrderedDict()
	for dist in unique_dist:
		dist_AB = []
		for AB in range(all_AB.shape[0]):
			if all_AB[AB,2] == dist:
				dist_AB.append(all_AB[AB,:])
		# Convert to array
		dist_AB = np.array(dist_AB)
		all_AB_dist_sorted[str(dist)] = dist_AB

	# Create all unidimensional analogies
	all_analogies = []
	for dist in unique_dist:
		dist_AB = all_AB_dist_sorted[str(dist)]
		for AB in range(dist_AB.shape[0]):
			for CD in range(dist_AB.shape[0]):
				if AB != CD:
					analogy = np.concatenate([dist_AB[AB,:-1], dist_AB[CD,:-1]])
					all_analogies.append(analogy)
	# Convert to array	
	all_analogies = np.array(all_analogies)

	# Generate cartesian product of features for irrelevant dimensions
	N_irrelevant_dimensions = config.N_dims - 1
	irrelevant_feature_prod = np.array([list(x) for x in product(range(config.N_levels_per_region), repeat=N_irrelevant_dimensions)])
	N_irrelevant_per_analogy = int(irrelevant_feature_prod.shape[0] * config.prcnt_full_space)

	#############################################################################################################################
	########################################### Feature values over full space ##################################################
	#############################################################################################################################

	log.info('Generating all feature values...')

	# Range of values for object location
	min_loc = int((config.img_size / 2.0) - (config.N_levels / 2.0))
	interval = 1
	obj_loc_vals = np.arange(min_loc, 
							 min_loc + (interval * config.N_levels),
							 interval)
	# Range of values for object size
	obj_size_vals = np.arange(config.min_obj_size, 
							  config.min_obj_size + ((interval * 2) * config.N_levels),
							  interval * 2)
	# Range of brightnesses
	obj_brightness_vals = np.linspace(config.min_obj_brightness, config.max_obj_brightness, config.N_levels).astype(np.int)

	# All feature values
	all_feature_vals = np.swapaxes(np.array([obj_loc_vals, obj_loc_vals, obj_size_vals, obj_brightness_vals]),0,1)

	#############################################################################################################################
	############################################## Create training/test sets ####################################################
	#############################################################################################################################

	log.info('Generating train/test sets...')

	# Number of regions
	N_regions = int(config.N_levels / config.N_levels_per_region)

	# Iterate over all regions
	for r in range(N_regions):

		# Create analogies in all dimensions
		all_analogy_objs = []
		all_analogy_dims = []
		all_analogy_ABCD = []
		all_analogy_dist = []
		all_analogy_not_D = []
		for d in range(config.N_dims):
			# Iterate over all unidimensional analogies
			for a in range(all_analogies.shape[0]):
				# Re-shuffle all irrelevant feature combinations
				irrelevant_ind = np.arange(irrelevant_feature_prod.shape[0])
				np.random.shuffle(irrelevant_ind)
				irrelevant_feature_prod = irrelevant_feature_prod[irrelevant_ind,:]
				# Iterate over irrelevant feature combinations
				for i in range(N_irrelevant_per_analogy):
					# Irrelevant feature values (tiled)
					irrelevant_feaures = np.tile(np.expand_dims(irrelevant_feature_prod[i,:],0), [config.N_levels_per_region, 1])
					# Combine with all possible values in relevant dimension
					analogy_objs = np.insert(irrelevant_feaures, d, np.arange(config.N_levels_per_region), axis=1)
					# Possible analogy objects
					all_analogy_objs.append(analogy_objs)
					# Relevant dimension
					all_analogy_dims.append(d)
					# Indices for [A, B, C, D] objects
					all_analogy_ABCD.append(all_analogies[a,:])
					# Analogy distance
					all_analogy_dist.append(all_analogies[a,1] - all_analogies[a,0])
					# Create list of indices for all objects that are NOT D
					D_ind = all_analogies[a,3]
					not_D = np.arange(config.N_levels_per_region)[np.arange(config.N_levels_per_region)!=D_ind]
					all_analogy_not_D.append(not_D)
		# Convert to arrays
		all_analogy_objs = np.array(all_analogy_objs)
		all_analogy_dims = np.array(all_analogy_dims)
		all_analogy_ABCD = np.array(all_analogy_ABCD)
		all_analogy_dist = np.array(all_analogy_dist)
		all_analogy_not_D = np.array(all_analogy_not_D)

		# Shuffle analogies
		N_analogies = all_analogy_objs.shape[0]
		analogy_ind = np.arange(N_analogies)
		np.random.shuffle(analogy_ind)
		all_analogy_objs = all_analogy_objs[analogy_ind, :, :]
		all_analogy_dims = all_analogy_dims[analogy_ind]
		all_analogy_ABCD = all_analogy_ABCD[analogy_ind, :]
		all_analogy_dist = all_analogy_dist[analogy_ind]
		all_analogy_not_D = all_analogy_not_D[analogy_ind, :]

		# Start/end indices
		region_start = r * config.N_levels_per_region
		region_end = (r+1) * config.N_levels_per_region
		# Feature values
		region_feature_vals = all_feature_vals[region_start:region_end, :]

		# Open file
		if r == 0:
			region_name = 'train'
		else:
			region_name = 'test' + str(r)
		dset_name = 'analogy_' + region_name
		filename = config.dir_name + '/' + dset_name + '.hy'
		f = h5py.File(filename, 'w')

		# Directory for drawing images
		check_path('../figures/')
		img_figs_dir = '../figures/' +  'analogy_' + region_name + '/'
		check_path(img_figs_dir)

		# Iterate over all analogies 
		log.info('    ' + dset_name + '...')
		checkpoint = 1000
		for a in range(N_analogies):

			# Create images
			all_imgs = []
			all_latent_class = []
			for o in range(config.N_levels_per_region):
				# Latent class
				latent_class = all_analogy_objs[a, o, :]
				all_latent_class.append(latent_class)
				# Feature values
				X = region_feature_vals[all_analogy_objs[a, o, 0], 0]
				Y = region_feature_vals[all_analogy_objs[a, o, 1], 1]
				size = region_feature_vals[all_analogy_objs[a, o, 2], 2]
				brightness = (0, region_feature_vals[all_analogy_objs[a, o, 3], 3], 0)
				# Image
				img = gen_image(config.img_size, config.bg_color, brightness, size, X, Y)
				all_imgs.append(img)
			# Convert to arrays
			all_imgs = np.array(all_imgs)
			all_latent_class = np.array(all_latent_class)

			# Create group
			id = '{}'.format(a)
			grp = f.create_group(id)

			# Add data to group
			grp['imgs'] = all_imgs
			grp['latent_class'] = all_latent_class
			grp['analogy_dim'] = all_analogy_dims[a]
			grp['ABCD'] = all_analogy_ABCD[a,:]
			grp['dist'] = all_analogy_dist[a]
			grp['not_D'] = all_analogy_not_D[a]

			# Checkpoint
			if a % checkpoint == 0 and a != 0:
				log.info('        ' + str(a) + ' of ' + str(N_analogies) + '...')
				draw_images(all_imgs, all_analogy_ABCD[a,:], img_figs_dir, a, all_analogy_dims[a])

		# Close file
		f.close()

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--prcnt_full_space', type=float, default=0.1)
	parser.add_argument('--N_levels', type=int, default=42)
	parser.add_argument('--N_levels_per_region', type=int, default=7)
	parser.add_argument('--N_dims', type=int, default=4)
	parser.add_argument('--img_size', type=int, default=128)
	parser.add_argument('--min_obj_size', type=int, default=3)
	parser.add_argument('--bg_color', type=tuple, default=(128, 128, 128))
	parser.add_argument('--min_obj_brightness', type=float, default=(255 * 0.4))
	parser.add_argument('--max_obj_brightness', type=float, default=255)

	args = parser.parse_args()

	args.dir_name = '../datasets'
	check_path(args.dir_name)

	# Generate datasets
	generator(args)

if __name__ == '__main__':
	main()