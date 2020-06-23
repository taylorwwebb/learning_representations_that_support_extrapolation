# Import modules
import tensorflow as tf
import numpy as np
import pdb
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

        # Get variable names to create input operations
        model_script_name = "model_" + self.model_name
        create_input_var_names = getattr(__import__(model_script_name, fromlist=["create_input_var_names"]), "create_input_var_names")
        input_var_names = create_input_var_names()

        ## Training set
        # Read file
        train_set_name = self.config.dset_names[self.config.train_set_ind]
        train_filename = 'analogy_' + train_set_name + '.hy'
        train_file = os.path.join(self.dataset_path, train_filename)
        log.info("Reading %s...", train_file)
        train_data = h5py.File(train_file, 'r')
        # Create dataset structure
        train_ids = np.arange(len(train_data))
        np.random.shuffle(train_ids)
        train_set = Dataset(train_ids, train_data, 'analogy_' + train_set_name, input_var_names)
        # Create input operation
        self.train_batch_ops = create_input_ops(train_set, int(self.config.batch_size), scope = 'analogy_' + train_set_name + '_inputs')

        # Import model class
        Model = getattr(__import__(model_script_name, fromlist=["Model"]), "Model")
        log.info("Using Model class: %s", self.model_name)

        # Create 'global_step' variable
        log.info("Initializing global step and optimizer...")
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)

        # Build model
        log.info("Building training model...")
        with tf.variable_scope(tf.get_variable_scope()):
            self.model = Model(self.config, self.train_batch_ops)
            tf.get_variable_scope().reuse_variables()
            self.train_loss = self.model.train_loss

        # Create optimizer
        log.info("Create optimizer...")
        self.opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.opt = hvd.DistributedOptimizer(self.opt)

        # Set up training operation (to apply gradients)
        log.info('Setting up training operation...')
        self.apply_gradient_op = self.opt.minimize(self.train_loss, global_step=self.global_step)

        # Model name (for output directory)
        self.model_name = self.config.model_name + '_trained_on_' + train_set_name + '_run' + self.config.run

        # Create directory for training output
        check_path('./train_dir')
        self.train_dir = './train_dir/%s' % (self.model_name) + '/' 
        check_path(self.train_dir)
        log.info("Train Dir: %s", self.train_dir)

        # Summaries and saving output
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=len(self.config.save_checkpoint_steps))

        # Set up session
        log.info("Setting up session...")
        init = tf.global_variables_initializer()
        bcast = hvd.broadcast_global_variables(0)
        self.session = tf.Session(config=self.session_config)
        self.session.run(init) 
        self.session.run(bcast)           
        tf.train.start_queue_runners(sess=self.session)

    def train(self):

        # Use CPU to coordinate training
        with tf.Graph().as_default(), tf.device('/cpu:0'):

            log.info("Training begins...")

            # Setup file for recording timecourses
            timecourse_dir = './timecourses'
            check_path(timecourse_dir)
            timecourse_filename = self.model_name + '.txt'
            timecourse_filename_path = os.path.join(timecourse_dir, timecourse_filename)
            timecourse_file_ID = open(timecourse_filename_path, 'w')
            header_line = 'step loss'
            for key in self.model.all_out.keys():
                header_line = header_line + ' ' + key
            header_line = header_line + '\n'
            timecourse_file_ID.write(header_line)

            train_steps = int(self.config.train_steps)
            save_checkpoint_steps = np.round(np.array(self.config.save_checkpoint_steps)).astype(np.int)
            for s in range(train_steps):

                ## Training
                # Single training step
                summary, step_time, train_loss, train_all_out = self.run_single_step(s)
                # Periodic output
                if s % 10 == 0:               
                    # Output performance
                    self.log_summary(s+1, step_time, train_loss, train_all_out)
                    # Write performance to timecourse file
                    timecourse_line = str(int(s+1)) + ' ' + str(train_loss)
                    for key in self.model.all_out.keys():
                        timecourse_line += ' ' + str(float("{0:.2f}".format(train_all_out[key])))
                    timecourse_line += '\n'
                    timecourse_file_ID.write(timecourse_line)

                # Save parameters
                if np.any(s+1 == save_checkpoint_steps):
                    log.info('Saving parameters...')
                    self.saver.save(self.session, self.train_dir, global_step=self.global_step)

            # Close timecourse file
            timecourse_file_ID.close()

    def run_single_step(self, step):

        # Start time
        start_time = time.time()

        # Read in batch and run step
        _, summary, train_loss, train_all_out = self.session.run(
            [self.apply_gradient_op, self.summary_op, 
            self.model.train_loss, self.model.all_out])

        # End time
        end_time = time.time()
        step_time = end_time - start_time

        return summary, step_time, train_loss, train_all_out

    def log_summary(self, step, step_time, train_loss, train_out): 

        if step_time == 0:
            step_time = 0.001

        # Step number
        if step == 'final':
            text = " [train step final] "
        else:
            text = " [train step {step:4d}] ".format(step = step)

        # Training loss
        text = text + "{var_name:s}: {var_val:.4f} ".format(var_name = 'train_loss', var_val = train_loss)

        # Training outputs
        for key, val in train_out.items():
            text = text + "{var_name:s}: {var_val:.4f} ".format(var_name = key, var_val = val)

        # Step time
        text = text + "({sec_per_batch:.3f} sec/batch)".format(sec_per_batch = step_time)

        # Print
        log.info(text)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='context_norm')
    parser.add_argument('--dset_names', nargs="*", type=str, default=['train', 'test1', 'test2', 'test3', 'test4', 'test5'])
    parser.add_argument('--train_set_ind', type=int, default=0)
    parser.add_argument('--run', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--save_checkpoint_steps', nargs="*", type=int, default=[100,200,500,1000,2000,10000])
    parser.add_argument('--learning_rate', type=float, default=5e-4)

    config = parser.parse_args()

    # Construct trainer
    trainer = Trainer(config)

    # Train
    trainer.train()

if __name__ == '__main__':
    main()
