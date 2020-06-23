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

class Evaler(object):

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

        # Create copy of model for each test set
        self.all_models = []
        for t in range(len(self.config.test_set_ind)):
            # Read file
            test_set_name = self.config.test_set_names[self.config.test_set_ind[t]]
            test_filename = 'analogy_' + test_set_name + '.hy'
            test_file = os.path.join(self.dataset_path, test_filename)
            log.info("Reading %s...", test_file)
            test_data = h5py.File(test_file, 'r')
            self.test_data_size = len(test_data)
            # Create dataset structure
            test_ids = np.arange(self.test_data_size)
            np.random.shuffle(test_ids)
            test_set = Dataset(test_ids, test_data, 'analogy_' + test_set_name, input_var_names)
            # Create input operation
            test_batch_ops = create_input_ops(test_set, int(self.config.batch_size), scope = 'analogy_' + test_set_name + '_inputs')
            # Build model
            log.info("Building model...")
            with tf.variable_scope(tf.get_variable_scope()):
                model = Model(self.config, test_batch_ops, is_train=False)
                self.all_models.append(model)

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

    def eval(self):

        # Use CPU to coordinate evaluation
        with tf.Graph().as_default(), tf.device('/cpu:0'):

            log.info("Begin evaluation...")

            # Directory for evaluation output
            eval_dir = './eval'
            check_path(eval_dir)
            model_eval_dir = eval_dir + '/' + self.model_name
            check_path(model_eval_dir)

            # Loop over all test sets
            for t in range(len(self.config.test_set_ind)):

                # Name of test set
                test_set_name = self.config.test_set_names[self.config.test_set_ind[t]]
                log.info('Evaluating model on dataset: ' + test_set_name + '...')

                eval_steps = int((self.test_data_size / self.config.batch_size))
                all_out = []
                for s in range(eval_steps):

                    # Single evaluation step
                    log.info('Step ' + str(s+1) + ' of ' + str(eval_steps) + '...')
                    out = self.session.run(self.all_models[t].all_out)
                    all_out.append(out)

                # Set up file for saving results
                eval_filename = model_eval_dir + '/' + test_set_name + '.txt'
                eval_file_ID = open(eval_filename, 'w')

                # Write header
                variable_names = [*all_out[0]]
                header = ''
                for v in range(len(variable_names)):
                    header += variable_names[v]
                    if v == len(variable_names) - 1:
                        header += '\n'
                    else:
                        header += ' '
                eval_file_ID.write(header)

                # Write results
                results = ''
                for v in range(len(variable_names)):
                    v_all_steps = []
                    for s in range(eval_steps):
                        v_all_steps.append(all_out[s][variable_names[v]])
                    v_mean = np.mean(v_all_steps)
                    results += str(v_mean)
                    if v != len(variable_names) - 1:
                        results += ' '
                eval_file_ID.write(results)

                # Close file
                eval_file_ID.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='context_norm')
    parser.add_argument('--test_set_names', nargs="*", type=str, default=['train', 'test1', 'test2', 'test3', 'test4', 'test5'])
    parser.add_argument('--test_set_ind', nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--train_set_ind', type=int, default=0)
    parser.add_argument('--run', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', type=int, default=10000)

    config = parser.parse_args()

    # Construct evaluation module
    evaler = Evaler(config)

    # Evaluate
    evaler.eval()

if __name__ == '__main__':
    main()
