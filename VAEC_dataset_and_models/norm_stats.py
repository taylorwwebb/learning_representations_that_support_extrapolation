# Import modules
import tensorflow as tf
import numpy as np
import argparse
import os
import time
import h5py
import sys
import matplotlib.pyplot as plt
import horovod.tensorflow as hvd

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

# Add models and tasks to path
sys.path.insert(0, './models')
sys.path.insert(0, './tasks')

# Import methods
from util import log
from input_ops import *

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


class Trainer(object):

    def __init__(self,config):

        # Initialize Horovod
        hvd.init()

        # Pin GPU to be used to process local rank (one GPU per process)
        self.session_config = tf.ConfigProto()
        self.session_config.gpu_options.visible_device_list = str(hvd.local_rank())
        self.session_config.gpu_options.allow_growth = True
        log.info('# of GPUs = ' + str(hvd.size()))

        # Config
        self.config = config
        self.config.N_GPUs = hvd.size()
        self.model_name = config.model_name

        # Get dataset path 
        self.dataset_path = './datasets'

        # Get variable names to create input operations for
        model_script_name = "model_" + self.model_name
        create_input_var_names = getattr(__import__(model_script_name, fromlist=["create_input_var_names"]), "create_input_var_names")
        input_var_names = create_input_var_names()

        # Import model class
        Model = getattr(__import__(model_script_name, fromlist=["Model"]), "Model")
        log.info("Using Model class: %s", self.model_name)

        # Read file
        train_set_name = self.config.test_set_names[self.config.train_set_ind]
        train_filename = 'analogy_' + train_set_name + '.hy'
        train_file = os.path.join(self.dataset_path, train_filename)
        log.info("Reading %s...", train_file)
        train_data = h5py.File(train_file, 'r')
        self.train_data_size = len(train_data)
        # Create dataset structure
        train_ids = np.arange(self.train_data_size)
        np.random.shuffle(train_ids)
        train_set = Dataset(train_ids, train_data, 'analogy_' + train_set_name, input_var_names)
        # Create input operation
        train_batch_ops = create_input_ops(train_set, int(self.config.batch_size), scope = 'analogy_' + train_set_name + '_inputs')
        # Build model
        log.info("Building model...")
        with tf.variable_scope(tf.get_variable_scope()):
            self.model = Model(self.config, train_batch_ops, is_train=True)

        # Model name (for output directory)
        train_set_name = self.config.test_set_names[self.config.train_set_ind]
        self.model_name = self.config.model_name + '_trained_on_' + train_set_name + '_run' + self.config.run
        # Directory with saved parameters
        self.train_dir = './train_dir/%s' % (self.model_name) + '/' 

        # Summaries and saving output
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()

        # Set up session
        log.info("Setting up session...")
        init = tf.global_variables_initializer()
        bcast = hvd.broadcast_global_variables(0)
        self.session = tf.Session(config=self.session_config)
        self.session.run(init) 
        self.session.run(bcast)           
        tf.train.start_queue_runners(sess=self.session)

        # Restore session
        log.info("Loading parameters from trained model...")
        checkpoint = self.train_dir + '-' + str(self.config.checkpoint) 
        self.saver.restore(self.session, checkpoint)

    def train(self):

        # Use CPU to coordinate training
        with tf.Graph().as_default(), tf.device('/cpu:0'):

            log.info("Getting normalization stats...")

            eval_steps = int((self.train_data_size / self.config.batch_size))
            all_A_latent = []
            all_B_latent = []
            all_C_latent = []
            all_D_latent = []
            all_foil_latent = []
            for s in range(eval_steps):

                # Single evaluation step
                log.info('Step ' + str(s+1) + ' of ' + str(eval_steps) + '...')
                batch_stats = self.session.run(self.model.all_out_stats)
                all_A_latent.append(batch_stats['A_latent'])
                all_B_latent.append(batch_stats['B_latent'])
                all_C_latent.append(batch_stats['C_latent'])
                all_D_latent.append(batch_stats['D_latent'])
                all_foil_latent.append(batch_stats['all_foil_latent'])

            # Convert to matrices
            all_A_latent = np.concatenate(all_A_latent)
            all_B_latent = np.concatenate(all_B_latent)
            all_C_latent = np.concatenate(all_C_latent)
            all_D_latent = np.concatenate(all_D_latent)
            all_foil_latent = np.concatenate(all_foil_latent)

            # Calculate normalization stats
            log.info("Calculating stats...")
            A_latent_mean = np.mean(all_A_latent, axis=0)
            A_latent_var = np.var(all_A_latent, axis=0)
            B_latent_mean = np.mean(all_B_latent, axis=0)
            B_latent_var = np.var(all_B_latent, axis=0)
            C_latent_mean = np.mean(all_C_latent, axis=0)
            C_latent_var = np.var(all_C_latent, axis=0)
            D_latent_mean = np.mean(all_D_latent, axis=0)
            D_latent_var = np.var(all_D_latent, axis=0)
            all_foil_latent_mean = []
            all_foil_latent_var = []
            for f in range(all_foil_latent.shape[1]):
            	all_foil_latent_mean.append((np.mean(all_foil_latent[:,f,:], axis=0)))
            	all_foil_latent_var.append((np.var(all_foil_latent[:,f,:], axis=0)))
            all_foil_latent_mean = np.array(all_foil_latent_mean)
            all_foil_latent_var = np.array(all_foil_latent_var)

            # Set up file for saving stats
            log.info("Saving stats...")
            stats_fname = self.train_dir + str(self.config.checkpoint) + '_stats.hy'
            stats_f = h5py.File(stats_fname, 'w')
            # Write stats
            grp = stats_f.create_group('norm_stats')
            grp['A_latent_mean'] = A_latent_mean
            grp['A_latent_var'] = A_latent_var
            grp['B_latent_mean'] = B_latent_mean
            grp['B_latent_var'] = B_latent_var
            grp['C_latent_mean'] = C_latent_mean
            grp['C_latent_var'] = C_latent_var
            grp['D_latent_mean'] = D_latent_mean
            grp['D_latent_var'] = D_latent_var
            grp['all_foil_latent_mean'] = all_foil_latent_mean
            grp['all_foil_latent_var'] = all_foil_latent_var

            # Close file
            stats_f.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='batch_norm_train_stats')
    parser.add_argument('--test_set_names', nargs="*", type=str, default=['train', 'test1', 'test2', 'test3', 'test4', 'test5'])
    parser.add_argument('--train_set_ind', type=int, default=0)
    parser.add_argument('--run', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', type=int, default=10000)

    config = parser.parse_args()

    # Construct trainer
    trainer = Trainer(config)

    # Train
    trainer.train()

if __name__ == '__main__':
    main()
