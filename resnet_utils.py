#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  resnet Structure Utils
  Created: 09/06/17
"""

import tensorflow as tf
import numpy as numpy

BN_EPSILON = 0.001
initializer=tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(0.0002)

def conv(input_layer, filter_shape, stride=1, regulation_decay=0.0002):
    '''
    Conv ECG

    :param input_layer:   4D Tensor:  [batch_num, ECG_height, ECG_duration, ECG_Depth]
    :param filter_shape:  1D List:    [filter_height, filter_duration, filter_depth, filter_number]
    :param stride:        stride size for conv
    :param regulation_decay: weight decay scale for l2 regularization [0.0002]

    :return a conved 4D tensor
    '''
    # initializer=tf.contrib.layers.xavier_initializer()
    # regularizer = tf.contrib.layers.l2_regularizer()

    filter_num = filter_shape[-1]
    filter = tf.get_variable(
      name='conv_filter',
      shape=filter_shape,
      initializer=initializer,
      regularizer=regularizer
    )
    conv_layer = tf.nn.conv2d(
      input=input_layer,
      filter=filter,
      strides=[1,stride,stride,1],
      padding='SAME'
    )

    return conv_layer

def batch_norm(input_layer, output_channel):
    '''
    batch normal current tensor.

    :params input_layer: 4D tensor
    :params output_channel: corresponding to filter_number

    :return batched layer
    '''
    mean,variance = tf.nn.moments(input_layer,axes=[0,1,2])
    beta = tf.get_variable(
      'beta',
      output_channel,
      tf.float32,
      initializer=tf.constant_initializer(0.0, tf.float32)
    )
    gamma = tf.get_variable(
      'gamma',
      output_channel,
      tf.float32,
      initializer=tf.constant_initializer(1.0, tf.float32)
    )
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def residual_block(input_layer, output_channel, filter_height, filter_duration, first_block=False, WTFstanford=True):
    '''
    Inspired from resnet.py
    URL: XXXX

    Defines a residual block in resnet

    :params input_layer: 4D tensor
    :params output_channel: corresponding to filter_number
    :params filter_height
    :params filter_duration
    :params first_block: if the input_layer is the first residual block of the whole network
    :params WFTstanford: if TRUE then we use stanford's resnet

    :return 4D tensor
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block == True:
            filter = tf.get_variable(
              name='conv_filter',
              shape=[filter_height, filter_duration, input_channel, output_channel],
              initializer=initializer,
              regularizer=regularizer
            )
            conv1 = tf.nn.conv2d(
              input = input_layer,
              filter= filter,
              strides= [1,1,1,1],
              padding='SAME'
            )
        elif first_block == False:
            bn1 = batch_norm(
              input_layer=input_layer,
              output_channel=input_channel
            )
            relu_layer = tf.nn.relu(bn1)
            filter = tf.get_variable(
              name='conv_filter',
              shape=[filter_height,filter_duration,input_channel,output_channel],
              initializer=initializer,
              regularizer=regularizer
            )
            conv1 = tf.nn.conv2d(
              input=relu_layer,
              filter=filter,
              strides=[1,stride,stride,1],
              padding='SAME'
            )
    with tf.variable_scope('conv2_in_block'):
        bn2 = batch_norm(
          input_layer=conv1,
          output_channel=output_channel
        )
        relu_layer = tf.nn.relu(bn2)
        filter = tf.get_variable(
          name='conv_filter',
          shape=[filter_height,filter_duration,output_channel,output_channel],
          initializer=initializer,
          regularizer=regularizer
        )
        conv2 = tf.nn.conv2d(
          input=relu_layer,
          filter=filter,
          strides=[1,stride,stride,1],
          padding='SAME'
        )

    # when the channels of input layer and conv2 does not match ,we add zero pads to increase the depth of the input layer
    if increase_dim == True:
        if WTFstanford == True:
            pooled_input = tf.nn.avg_pool(
              value=input_layer,
              ksize=[1,2,2,1],
              strides=[1,2,2,1],
              padding='VALID'
            )
        elif WTFstanford == False:
            pooled_input = tf.nn.max_pool(
              value=input_layer,
              ksize=[1,2,2,1],
              strides=[1,2,2,1],
              padding='VALID'
            )
        padded_input = tf.pad(
          tensor=pooled_input,
          paddings=[[0,0],[0,0],[0,0],[input_channel//2,input_channel//2]]
        )
    elif increase_dim == False:
          padded_input = input_layer

    # add together
    block_output = conv2+padded_input
    return block_output

def linear(input_layer, num_labels):
    '''
    linear input tensor into X, and use wX+b to output

    :param input_layer: 2D tensor
    :param num_labels: total labels numbers

    :return fc_output Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = tf.get_variable(
      name='fc_w',
      shape=[input_dim,num_labels],
      initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
      regularizer=regularizer
    )
    fc_b = tf.get_variable(
      name='fc_b',
      shape=[num_labels],
      initializer=tf.zeros_initializer()
    )

    fc_output = tf.matmul(input_layer,fc_w)+fc_b
    return fc_output

