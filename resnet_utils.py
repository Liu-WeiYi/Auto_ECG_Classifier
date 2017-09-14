#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  resnet Structure Utils
  Created: 09/06/17
  Updated:  09/13/17 --- conv1d
"""

import tensorflow as tf
import numpy as numpy
import sys

BN_EPSILON = 0.001
# Define Global initializer and regularizer for global optimization :)
INITIALIZER =tf.contrib.layers.xavier_initializer()
REGULARIZER = tf.contrib.layers.l2_regularizer(0.0002)

def conv(input_layer, filter_shape, stride=1):
    '''
    Conv ECG

    :param: input_layer   3D Tensor  [batch_num, ECG_duration, ECG_Depth]
    :param: filter_shape  3D Tensor  [filter_duration, filter_depth, filter_number]
    :param: stride        stride size for conv

    :return a conved 3D tensor
    '''
    filter = tf.get_variable(
      name='conv_filter',
      shape=filter_shape,
      initializer=INITIALIZER,
      regularizer=REGULARIZER
    )
    conv_layer = tf.nn.conv1d(
      value=input_layer,
      filters=filter,
      stride=stride,
      padding='SAME'
    )

    return conv_layer

def batch_norm(input_layer, output_channel):
    '''
    batch normal current tensor.

    :params input_layer: 3D tensor
    :params output_channel: corresponding to filter_number

    :return batched layer
    '''
    # weild sentences... We should use higher level tf functions
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

    # bn_layer = tf.contrib.layers.batch_norm(input_layer)
    return bn_layer

def residual_block(input_layer, output_channel, filter_duration, first_block=False, WTFstanford=True):
    '''
    Inspired from residual block from resnet.py

    Defines a residual block in resnet

    :params input_layer: 3D tensor [batch, in_width(duration), in_channels]
    :params output_channel: corresponding to filter_number
    :params filter_duration
    :params first_block: if the input_layer is the first residual block of the whole network
    :params WFTstanford: if TRUE then we use stanford's resnet(use maxpooling)

    :return 3D tensor
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When shrinked the image size, we use stride = 2
    # from resnet.py
    if input_channel * 2 == output_channel:
        # works for original res
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        # works for Stanford res
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')
        sys.exit()

    # Residual Block Conv layer: BN->ReLu[->dropout]->conv
    # Only first residual block Conv layer does not have BN and ReLu!!
    with tf.variable_scope('conv1_in_block'):
        if first_block == True:
            # first residual block starts with conv layer.
            filter_shape = [filter_duration, input_channel, output_channel]
            conv1 = conv(input_layer,filter_shape,stride)
        elif first_block == False:
            bn1 = batch_norm(
              input_layer=input_layer,
              output_channel=input_channel
            )
            relu_layer = tf.nn.relu(bn1)
            filter_shape = [filter_duration, input_channel, output_channel]
            conv1 = conv(relu_layer, filter_shape, stride)
    with tf.variable_scope('conv2_in_block'):
        bn2 = batch_norm(
          input_layer=conv1,
          output_channel=output_channel
        )
        relu_layer = tf.nn.relu(bn2)
        filter_shape = [filter_duration, output_channel, output_channel]
        conv2 = conv(input_layer,filter_shape,stride)

    # Reuse input through avg_pooling / max_pooling
    if increase_dim is True:
        if self.WTFstanford == True:
            pooled_input = tf.layers.average_pooling1d(
              inputs = input_layer,
              pool_size = 2,
              strides = 2,
              padding='valid'
            )
        elif self.WTFstanford == False:
            pooled_input = tf.layers.max_pooling1d(
              inputs = input_layer,
              pool_size = 2,
              strides = 2,
              padding='valid'
            )
        # TODO: FIXME: How to pad... gosh...
        # padded_input = tf.pad() ??? <--- PAD到底怎么用哦！麻蛋！！
    elif increase_dim is False:
        padded_input = input_layer

    """
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
    """

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
      regularizer=REGULARIZER
    )
    fc_b = tf.get_variable(
      name='fc_b',
      shape=[num_labels],
      initializer=tf.zeros_initializer()
    )

    fc_output = tf.matmul(input_layer,fc_w)+fc_b
    return fc_output

