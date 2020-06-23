import numpy as np
import tensorflow as tf
from util import log

class Dataset(object):

    def __init__(self, ids, data, name, input_var_names):

        self.ids = list(ids.astype(str))
        self.name = name
        self.data = data  
        self.input_var_names = input_var_names   

    def get_data(self, id):

        ## Data retrieval
        inputs = []
        for var in range(len(self.input_var_names)):
            inputs.append(self.data[id][self.input_var_names[var]].value.astype(np.float32))

        return self.input_var_names, inputs

def create_input_ops(dataset,
                     batch_size,
                     scope='train_inputs',
                     allow_smaller_final_batch=False):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''
    input_ops = {}
    log.info("train_input_ops [%s]: Using %d IDs from dataset", scope, len(dataset.ids))

    # single operations
    with tf.device("/cpu:0"), tf.name_scope(scope):
        
        input_ops['id'] = tf.train.string_input_producer(
           tf.convert_to_tensor(dataset.ids), capacity=128
        ).dequeue(name='train_input_ids_dequeue')

        input_var_names, inputs = dataset.get_data(dataset.ids[0])

        def load_fn(id):
            _, inputs = dataset.get_data(id)
            inputs.insert(0,id)
            return inputs

        T_out_list = [tf.string]
        for var in range(len(inputs)):
            T_out_list.append(tf.float32)

        input_ops_list = tf.py_func(
            load_fn, inp=[input_ops['id']],
            Tout=T_out_list,
            name='func'
        )

        input_ops['id'] = input_ops_list[0]
        for var in range(len(inputs)):
            input_ops[input_var_names[var]] = input_ops_list[var+1]

        input_ops['id'].set_shape([])
        for var in range(len(inputs)):
            input_ops[input_var_names[var]].set_shape(list(inputs[var].shape))

    # batchify
    capacity = 2 * batch_size
    min_capacity = int(capacity * 0.75)
    batch_ops = tf.train.shuffle_batch(input_ops, 
        batch_size=batch_size, capacity=capacity, min_after_dequeue=min_capacity, allow_smaller_final_batch=allow_smaller_final_batch)

    return batch_ops

