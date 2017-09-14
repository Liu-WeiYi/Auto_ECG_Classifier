#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Auto ECG Classifier
  Created: 09/06/17
  Updated:  09/13/17 --- conv1d
"""

import tensorflow as tf
from datetime import datetime
from math import ceil

import resnet_utils as utils
from ECG_Data_Input import *


class ECG_Classifier:
    '''
    ECG Classifier
    '''
    # -----------------------------
    def __init__(self, args):
        # Model Selection
        self.WTFstanford        = args.original_res
        # Model Version
        self.version            = args.version
        # Model Check Point
        self.use_ckpt           = args.use_ckpt
        self.ckpt_path          = args.ckpt_path
        # resnet Related
        self.train_batch_size   = args.train_batch_size # default [20]
        self.epoch              = args.epoch            # default [10000]
        self.init_lr            = args.init_lr          # default [0.1]
        self.lr_decay_value     = args.lr_decay_rate    # default [0.96]
        self.lr_decay_step      = args.lr_decay_step    # default [10000]
        self.regulation_decay   = args.regularization_weight_decay
        self.residual_block_num = args.residual_block_num
        self.filter_Duration    = args.filter_Duration  # default [100]
        self.filter_Depth       = args.filter_Depth
        self.filter_init        = args.init_f_channels  # default [32]
        self.keep_prob          = args.keep_prob        # default [0.5]
        self.inputECG_Duration  = args.inputECG_Duration# default [3000]
        self.inputECG_Depth     = args.inputECG_Depth
        # data related
        self.load_data_to_mem   = args.load_all_data_to_memory
        self.num_train_data     = args.num_train_data
        self.num_labels         = args.num_labels
        self.num_test_data      = args.num_test_data

        # Define Place Holders
        """
        For Conv1d:
          training Sample:  [batch, in_width, in_channels]
          training filter:  [filter_width, in_channels, out_channels]
        """
        # 1. Placeholder for Training
        self.ECG_train_placeholder = tf.placeholder(
          dtype=tf.float32,
          shape=[self.train_batch_size, self.inputECG_Duration, self.inputECG_Depth]
        )
        self.ECG_train_labels_placeholder = tf.placeholder(
          dtype=tf.int32,
          shape=[self.train_batch_size]
        )
        # 2. Placeholder for Testing
        self.ECG_test_placeholder = tf.placeholder(
          dtype = tf.float32,
          shape=[self.num_test_data, self.inputECG_Duration, self.inputECG_Depth]
        )
        self.ECG_test_labels_placeholder = tf.placeholder(
          dtype=tf.int32,
          shape=[self.num_test_data]
        )

    # ----------------------------- #
    # Public Funcs
    # modelConstruction()
    # train()
    # test()
    # ----------------------------- #
    def modelConstruction(self, input_tensor_batch, reuseModel=False):
        '''
        Construct model with input_tensor.

        :param: input_tensor_batch [batch_size, ECG_Duration, ECG_Depth]
        :param: reuseModel False We need to construct the model.
          --- 1. False We need to construct the model.
          --- 2. True We can reuse the constructed model.

        :return: logits: return constructed model output
        '''
        print('======================================================')
        print('Model Construction...')

        logits = self.__resnet_structure(input_tensor_batch, reuse=reuseModel)
        return logits

    # -----------------------------
    def train(self):
        print('======================================================')
        print('Training Process Begin...')
        # 1. Construct Model
        logits = self.modelConstruction(
          input_tensor_batch = self.ECG_train_placeholder,
          reuseModel=False
        )
        # 2. Define Loss and Regulation Loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        model_loss = self.__loss(logits, self.ECG_train_labels_placeholder)
        '''
        Please note that
        regu_losses is a LIST object, and len(regu_losses) = 32 [the conv times (depth) of a resnet 32]
        model_loss.shape = []
        Hence, we need to add n (32) times of the model_loss and regu_losses
        '''
        # ∵ model_loss.shape = 1
        # ∴ [model_loss] + regu_losses is needed!
        loss = tf.add_n([model_loss]+regu_losses)

        global_step = tf.Variable(initial_value=0, trainable=False)
        # 3. Define learning rate decay
        decayed_lr = tf.train.exponential_decay(
          learning_rate = self.init_lr,
          global_step=global_step,
          decay_steps=self.lr_decay_step,
          decay_rate=self.lr_decay_value
        )

        # 4. Define optimization function
        opt = tf.train.MomentumOptimizer(
          learning_rate=decayed_lr,
          momentum=0.9
        ).minimize(
          loss=loss,
          global_step=global_step
        )

        print('======================================================')
        print('-- Total training variables num:',len(tf.trainable_variables()))
        print('Training Process Init...')

        init = tf.global_variables_initializer()
        sess = tf.Session()

        # Load from a checkpoint
        if self.use_ckpt is True:
            saver = tf.train.Saver(tf.global_variables())
            print('Use pre-trained model from %s'%self.ckpt_path)
            saver.restore(sess, self.ckpt_path)
        else:
            sess.run(init)

        print('======================================================')
        print('Training Process Start...')
        all_train_data = None
        all_train_labels = None
        if self.load_data_to_mem is True:
            print('-- Load all data into memory')
            all_train_data,all_train_labels = load_all_train_data_to_memory()
            if all_train_data != None and all_labes != None:
                assert all_train_data.shape[0] == len(all_train_labels)
        else:
            #TODO: load data seperately
            pass

        trainable_data_steps = ceil(self.num_train_data/self.train_batch_size)

        for epoch in range(self.epoch):
            for offset in range(trainable_data_steps):
                # load batch ECGs and corresponding batch labels for ECGs
                batch_ECGs, batch_labels = prepare_train_data(
                  offset = offset,
                  batch_size = self.train_batch_size,
                  total_num_data = self.num_train_data,
                  all_data = all_train_data,
                  all_labels=all_train_labels
                )
                # train current batch
                _, current_loss, current_lr, accuracy, step = sess.run(
                  fetches=[
                    opt,
                    loss,
                    decayed_lr,
                    self.__accuracy(batch_labels,logits),
                    global_step
                    ],
                  feed_dict={
                    self.ECG_train_placeholder:batch_ECGs,
                    self.ECG_train_labels_placeholder:batch_labels,
                  }
                )
            print('%s, Epoch[%d]: Current learning rate %.3f, loss %.8f, accuracy %.3f%%' %(datetime.time(),epoch+1, learning_rate, loss, accuracy*100))


    # -----------------------------
    def test(self, ckpt_path=''):
        print('======================================================')
        print('Testing Process Begin...')

        init = tf.global_variables_initializer()
        sess = tf.Session()
        # 0. load trained model
        assert ckpt_path != ''
        saver = tf.train.Saver(tf.global_variables())
        print('-- Use Trained model from %s'%self.ckpt_path)
        saver.restore(sess, self.ckpt_path)

        # 1. reuse constructed model
        logits = self.modelConstruction(
          input_tensor_batch = self.ECG_test_placeholder,
          reuseModel=True
        )

        # 2. load test data
        all_test_data, all_test_label = load_all_test_data_to_memory()
        assert all_test_data.shape[0] == len(all_test_label)

        # 3. evaluate test accuracy
        test_accuracy = sess.run(
          fetches=[self.__accuracy(all_test_label,logits)],
          feed_dict={
            self.ECG_test_placeholder:all_test_data,
            self.ECG_test_labels_placeholder:all_test_label
          }
        )
        print('Test Accuray is :\t%.3f'%test_accuracy)






    # ----------------------------- #
    # Private Funcs
    # __resnet_structure(input_tensor_batch, reuse=False)
    # __loss(logits, label_placeholder)
    # __accuracy(logits, labels)
    # ----------------------------- #
    def __resnet_structure(self, input_tensor_batch, reuse=False):
        '''
        The main function that defines the ResNet.
          * original resnet:  total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
          * standford resnet: total layers = 1 + 15*2 = 30

        :param: input_tensor_batch 3D tensor to input
          + For train: ECG_placeholder;
          + For test:  test_placeholder
        :param reuse To build train graph, reuse=False.
          + build test graph, reuse=False
          + build validation graph and share weights with train graph, resue=True

        :return: last layer in the network. WITH-NO-SOFTMAX!!!
        '''
        layers = []

        # pre-define filter output number
        filter_number = self.filter_init
        # construct first conv layer: conv->bn->relu
        with tf.variable_scope('conv0', reuse=reuse):
            # conv layer
            conv0 = utils.conv(
              input_layer=input_tensor_batch,
              filter_shape=[self.filter_Duration,self.filter_Depth,filter_number],
              stride=1,
            )
            bn0 = utils.batch_norm(
              input_layer=conv0,
              output_channel=filter_number
            )
            conv0_bn0_ReLU0 = tf.nn.relu(bn0)
            layers.append(conv0_bn0_ReLU0)

        if self.WTFstanford == True:
            # we use original resnet instead
            print('-- We choose to use Original ResNet!!')
            ''' 1. First Residual Model '''
            for i in range(self.residual_block_num):
                with tf.variable_scope('conv1_%d'%i, reuse=reuse):
                    if i == 0:
                        conv1 = utils.residual_block(
                          input_layer = layers[-1],
                          output_channel = filter_number,
                          filter_height = self.filter_Height,
                          filter_duration = self.filter_Duration,
                          first_block=True
                        )
                    else:
                        conv1 = utils.residual_block(
                          input_layer = layers[-1],
                          output_channel = filter_number,
                          filter_height = self.filter_Height,
                          filter_duration = self.filter_Duration
                        )
                    layers.append(conv1)

            ''' 2. Second Residual Model '''
            filter_number = 32
            for i in range(self.residual_block_num):
                with tf.variable_scope('conv2_%d'%i,reuse=reuse):
                    conv2 = utils.residual_block(
                      input_layer = layers[-1],
                      output_channel = filter_number,
                      filter_height = self.filter_Height,
                      filter_duration = self.filter_Duration
                    )
                    layers.append(conv2)

            ''' 3. Third Residual Model '''
            filter_number = 64
            for i in range(self.residual_block_num):
                with tf.variable_scope('conv3_%d'%i,reuse=reuse):
                    conv3 = utils.residual_block(
                      input_layer = layers[-1],
                      output_channel = filter_number,
                      filter_height = self.filter_Height,
                      filter_duration = self.filter_Duration
                    )
                layers.append(conv3)

        elif self.WTFstanford == False:
            ''' repeat residual block 15 times '''
            print('-- We choose to use Stanford ResNet!!')
            for i in range(self.residual_block_num*3+1):
                with tf.variable_scope('conv_layer_%d'%i, reuse=reuse):
                    if i == 0:
                        # first residual block starts from conv layer
                        conv = utils.residual_block(
                          input_layer=layers[-1],
                          output_channel=filter_number,
                          filter_duration = self.filter_Duration,
                          first_block=True,
                          WTFstanford=False
                        )
                    else:
                        # else start from BN layer
                        conv = utils.residual_block(
                          input_layer=layers[-1],
                          output_channel=filter_number,
                          filter_duration=self.filter_Duration,
                          first_block=False,
                          WTFstanford=False
                        )
                layers.append(conv)

        ''' 4. Fully Connected Layer '''
        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            fc_bn = utils.batch_norm(
              input_layer = layers[-1],
              output_channel = in_channel
            )
            fc_relu_layer = tf.nn.relu(fc_bn)
            # global_pool = tf.reduce_mean(fc_relu_layer,[1,2])
            global_pool = tf.reduce_mean(fc_relu_layer,axis=[1])

            fc_output = utils.linear(
              input_layer=global_pool,
              num_labels = self.num_labels
            )

            layers.append(fc_output)

        return layers[-1]

    # -----------------------------
    def __loss(self, logits, labels):
        '''
        Inspired by resnet.py
        Calculate Cross Entropy Loss given logits and true labels

        :param logtis: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]

        :return loss tensor with shape [1]
        '''
        labes = tf.cast(labels, tf.int64) # 转换成int64的格式
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits,
          labels=labels,
          name='cross_entropy_per_image'
        )
        cross_entropy_mean = tf.reduce_min(
          input_tensor=cross_entropy,
          name='average_cross_entropy'
        )
        return cross_entropy_mean

    def __accuracy(self, labels, logits):
        """tf.Sesssion运行该操作即可计算预测结果相比于label的准确率"""
        accuracy = tf.equal(labels, tf.cast(tf.argmax(logits, 1), tf.int32))
        return tf.reduce_mean(tf.cast(accuracy, tf.float32))