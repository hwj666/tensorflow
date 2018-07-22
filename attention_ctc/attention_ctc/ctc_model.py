# -*- coding: utf-8 -*-

import tensorflow as tf
# Layer params:   Filts K  Padding  Name     BatchNorm?
layer_params = [ [  64, 3, 'valid', 'conv1', True], 
                 [  64, 3, 'same',  'conv2', True], # pool
                 [ 128, 3, 'same',  'conv3', True], 
                 [ 128, 3, 'same',  'conv4', True], # hpool
                 [ 256, 3, 'same',  'conv5', True],
                 [ 256, 3, 'same',  'conv6', True], # hpool
                 [ 512, 3, 'same',  'conv7', True], 
                 [ 512, 3, 'same',  'conv8', True]] # hpool 3

rnn_size = 2**9
dropout_rate = 0.5

def conv_layer(bottom, params, training ):
    """Build a convolutional layer using entry from layer_params)"""

    batch_norm = params[4] # Boolean

    if batch_norm:
        activation=None
    else:
        activation=tf.nn.relu

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    top = tf.layers.conv2d(bottom, 
                           filters=params[0],
                           kernel_size=params[1],
                           padding=params[2],
                           activation=activation,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           name=params[3])
    if batch_norm:
        top = norm_layer( top, training, params[3]+'/batch_norm' )
        top = tf.nn.relu( top, name=params[3]+'/relu' )

    return top

def pool_layer( bottom, wpool, padding, name ):
    """Short function to build a pooling layer with less syntax"""
    top = tf.layers.max_pooling2d( bottom, 2, [2,wpool], 
                                   padding=padding, 
                                   name=name)
    return top

def norm_layer( bottom, training, name):
    """Short function to build a batch normalization layer with less syntax"""
    top = tf.layers.batch_normalization( bottom, axis=3, # channels last,
                                         training=training,
                                         name=name )
    return top


def convnet_layers(inputs, training):
    """Build convolutional network layers attached to the given input tensor"""
    # inputs should have shape [ ?, 32, ?, 1 ]
    with tf.variable_scope("convnet"): # h,w
        
        conv1 = conv_layer(inputs, layer_params[0], training ) # 30,30
        conv2 = conv_layer( conv1, layer_params[1], training ) # 30,30
        pool2 = pool_layer( conv2, 2, 'valid', 'pool2')        # 15,15
        conv3 = conv_layer( pool2, layer_params[2], training ) # 15,15
        conv4 = conv_layer( conv3, layer_params[3], training ) # 15,15
        pool4 = pool_layer( conv4, 1, 'valid', 'pool4' )       # 7,14
        conv5 = conv_layer( pool4, layer_params[4], training ) # 7,14
        conv6 = conv_layer( conv5, layer_params[5], training ) # 7,14
        pool6 = pool_layer( conv6, 1, 'valid', 'pool6')        # 3,13
        conv7 = conv_layer( pool6, layer_params[6], training ) # 3,13
        conv8 = conv_layer( conv7, layer_params[7], training ) # 3,13
        pool8 = tf.layers.max_pooling2d( conv8, [3,1], [3,1], 
                                   padding='valid', name='pool8') # 1,13
        features = tf.squeeze(pool8, axis=1, name='features') # [batch,max_steps,n_features]
        batch_size = tf.shape(features)[0]
        seq_length = tf.shape(features)[1]
        sequence_length = tf.fill([batch_size],seq_length)
        return features,sequence_length

def rnn_layer(bottom_sequence,rnn_size,scope):
    """Build bidirectional (concatenated output) RNN layer"""

    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    
    # Default activation is tanh
    cell_fw = tf.nn.rnn_cell.LSTMCell( rnn_size, 
                                       initializer=weight_initializer)
    cell_bw = tf.nn.rnn_cell.LSTMCell( rnn_size, 
                                       initializer=weight_initializer)
    
    # Include?
    #cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw, 
    #                                         input_keep_prob=dropout_rate )
    #cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw, 
    #                                         input_keep_prob=dropout_rate )
    
    rnn_output,_ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        dtype=tf.float32,
        scope=scope)
    
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    rnn_output_stack = tf.concat(rnn_output,2,name='output_stack')
    
    return rnn_output_stack


def rnn_layers(features, num_classes):
    """Build a stack of RNN layers from input features"""

    # Input features is [batchSize paddedSeqLen numFeatures]
    logit_activation = tf.nn.relu
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope("rnn"):
        # Transpose to time-major order for efficiency
        rnn1 = rnn_layer(features, rnn_size, 'bdrnn1')
        rnn2 = rnn_layer(rnn1, rnn_size, 'bdrnn2')
        rnn_logits = tf.layers.dense( rnn2, num_classes+1, 
                                      activation=logit_activation,
                                      kernel_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      name='logits')
        return rnn_logits

def ctc_loss_layer(rnn_logits, sequence_labels, sequence_length):
    """Build CTC Loss layer for training"""
    loss = tf.nn.ctc_loss( sequence_labels, rnn_logits, sequence_length,
                           time_major=False )
    total_loss = tf.reduce_mean(loss)
    return total_loss
