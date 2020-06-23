import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from util import log

# 2D convolutional net
def conv2d(input, kernel_sizes, N_channels, stride_sizes, padding='same',
            activation_fn=tf.nn.relu, bias_val=0.0, scope='conv_2D', reuse=True, latent_activation_fn=tf.nn.relu, latent_bias_val=0.0):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        for l in range(len(N_channels)):
            layer_name = 'layer_' + str(l + 1)    
            # Use sigmoid for final layer
            if l == (len(N_channels) - 1):
                activation_fn = latent_activation_fn
                bias_val = latent_bias_val
            with tf.variable_scope(layer_name) as scope:
                output = tf.layers.conv2d(input, int(N_channels[l]), int(kernel_sizes[l]), int(stride_sizes[l]), 
                    padding='same', activation=activation_fn, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                    bias_initializer=tf.constant_initializer(bias_val))
                input = output
    return output

# Transpose of 2D convolutional net
def conv2d_transpose(input, kernel_sizes, N_channels, stride_sizes, padding='same',
            activation_fn=tf.nn.relu, bias_val=0.0, scope='conv_2D', reuse=True, output_activation_fn=tf.nn.sigmoid, output_bias_val=0.0):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        for l in range(len(N_channels)):
            layer_name = 'layer_' + str(l + 1)
            # Use sigmoid for final layer
            if l == (len(N_channels) - 1):
                activation_fn = output_activation_fn
                bias_val = output_bias_val
            with tf.variable_scope(layer_name) as scope:
                output = tf.layers.conv2d_transpose(input, int(N_channels[l]), int(kernel_sizes[l]), int(stride_sizes[l]), 
                    padding='same', activation=activation_fn, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                    bias_initializer=tf.constant_initializer(bias_val))
                input = output
    return output

# Linear layer
def linear_layer(input, output_shape, scope="linear", reuse=True, bias_value=[]):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        with tf.device('/device:CPU:0'):
            w = tf.get_variable('w', [input.get_shape()[-1], output_shape], 
                initializer=tf.contrib.layers.xavier_initializer())
            if len(bias_value)==0:
                biases = tf.get_variable('biases', [output_shape],
                    initializer=tf.zeros_initializer())
            else:
                bias_value = np.tile(np.array(bias_value[0]),output_shape).astype(np.float32)
                biases = tf.get_variable('biases', initializer=bias_value, dtype=tf.float32)
        with tf.variable_scope("output"): output = tf.matmul(input, w) + biases
    return output, w, biases

# Generic fully-connected layer w/ optional batch normalization
def fc_layer(input, output_shape, batch_norm=False, activation_fn=tf.nn.relu, scope="fc", reuse=True, bias_value=[]):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        with tf.device('/device:CPU:0'):
            w = tf.get_variable('w', [input.get_shape()[-1], output_shape], 
                initializer=tf.contrib.layers.xavier_initializer())
            if len(bias_value)==0:
                biases = tf.get_variable('biases', [output_shape],
                    initializer=tf.zeros_initializer())
            else:
                bias_value = np.tile(np.array(bias_value[0]),output_shape).astype(np.float32)
                biases = tf.get_variable('biases', initializer=bias_value, dtype=tf.float32)
        linear_out = tf.matmul(input, w) + biases
        if batch_norm==True:
            linear_out_mn, linear_out_var = tf.nn.moments(linear_out,[0])
            bn_offset = biases = tf.get_variable(
                'bn_offset', [output_shape], initializer=tf.zeros_initializer())
            bn_scale = biases = tf.get_variable(
                'bn_scale', [output_shape], initializer=tf.ones_initializer())
            var_epsilon = 1e-3  # this is just a small number to avoid division by 0
            linear_out = tf.nn.batch_normalization(
                linear_out, linear_out_mn, linear_out_var, bn_offset, bn_scale, var_epsilon)
        output = activation_fn(linear_out)
    return output, w, biases

# MLP
def mlp(input, mlp_size, batch_norm=False, scope='mlp', reuse=True, activation_fn=tf.nn.relu):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        w_all_layers = []
        bias_all_layers = []
        for l in range(len(mlp_size)):
            layer_name = 'layer_' + str(l + 1)
            if l == 0:
                output, w, biases = fc_layer(input, mlp_size[l], batch_norm=batch_norm, scope=layer_name, reuse=reuse, activation_fn=activation_fn)
            else:
                output, w, biases = fc_layer(output, mlp_size[l], batch_norm=batch_norm, scope=layer_name, reuse=reuse, activation_fn=activation_fn)
            w_all_layers.append(w)
            bias_all_layers.append(biases)
        return output, w_all_layers, bias_all_layers

# LSTM 
def lstm(all_inputs, lstm_size, scope="lstm", reuse=True):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        # LSTM
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # Initialize cell state
        batch_size = int(all_inputs.shape[1])
        state = lstm.zero_state(batch_size, dtype=tf.float32)
        # Pass inputs to LSTM
        T = int(all_inputs.shape[0])
        all_output = []
        for t in range(T):
            output, state = lstm(all_inputs[t, :, :], state)
            all_output.append(output)
        # Stack output
        all_output = tf.stack(all_output, axis=0)
        return all_output

# LSTM w/ layer norm.
def lstm_layernorm(all_inputs, lstm_size, scope="lstm", reuse=True):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        # LSTM
        lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_size)
        # Initialize cell state
        batch_size = int(all_inputs.shape[1])
        state = lstm.zero_state(batch_size, dtype=tf.float32)
        # Pass inputs to LSTM
        T = int(all_inputs.shape[0])
        all_output = []
        for t in range(T):
            output, state = lstm(all_inputs[t, :, :], state)
            all_output.append(output)
        # Stack output
        all_output = tf.stack(all_output, axis=0)
        return all_output

# Loss and Accuracy 
def build_cross_entropy_loss(logits, labels):
    # Cross-entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # Correct predictions
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32)
    # Determine whether all logits are equal (along choice dimension)
    equal = []
    for c in range(1,int(logits.shape[1])):
        equal.append(tf.equal(logits[:,0],logits[:,c]))
    equal = tf.stack(equal, axis=1)
    not_all_equal = tf.cast(tf.reduce_any(tf.logical_not(equal), axis=1), tf.float32)
    # Do not count prediction as correct if all logits are equal
    correct_prediction = correct_prediction * not_all_equal
    # Overall accuracy
    accuracy = tf.reduce_mean(correct_prediction)
    return tf.reduce_mean(loss), accuracy, correct_prediction
