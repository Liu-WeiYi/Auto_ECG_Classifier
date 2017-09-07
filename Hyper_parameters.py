#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  设置 hyper_parameters
  Created: 09/06/17
"""
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()

    '''
    Model Selection
    '''
    parser.add_argument('--original_res', type=bool, default=False,
        help="use original resnet or not. [False]"
    )
    '''
    Model Version
    '''
    parser.add_argument("--version", type = str, default="ECG_Test",
        help="Define a specific version to defining the directory and to save logs and checkpoints"
    )
    '''
    Model Check Point
    '''
    parser.add_argument('--use_ckpt', type=bool, default=False,
        help="Use pre-trained model"
    )
    parser.add_argument('--ckpt_path', type=str, default='ECG_Test_log/',
        help="Check point directory to restore"
    )
    '''
    Model Train/Test Flag
    '''
    parser.add_argument('--train_flag', type=bool, default=False,
        help="Train Model"
    )

    parser.add_argument('--test_flag', type=bool, default=False,
        help="Test Accuracy"
    )
    '''
    resnet Related
    '''
    # 1. Training/Testing Related Hyper_parameters
    parser.add_argument('--epoch', type=int, default=10000,
        help="Total Training Epochs Number [10000]"
    )
    parser.add_argument('--train_batch_size', type=int, default=20,
        help="Training Batch Size [20]"
    )
    parser.add_argument('--test_batch_size', type=int, default=20,
        help="Testing Batch Size [20]"
    )
    parser.add_argument('--init_lr', type=float, default=0.1,
        help="initial learning rate [0.1]"
    )
    parser.add_argument('--lr_decay_value', type=float, default=0.1,
        help="learning rate decay value [0.1]"
    )
    parser.add_argument('--lr_decay_step', type=int, default=5000,
        help="learning rate decay step [5000]"
    )
    parser.add_argument('--regularization_weight_decay', type=float, default=0.0002,
        help="weight decay scale for l2 regularization [0.0002]"
    )
    # 2. resnet Related Hyper_parameters
    parser.add_argument('--residual_block_num', type=int, default=5,
        help="residual block num. [5]"
    )
    parser.add_argument('--filter_Duration', type=int, default=2,
        help="filter duration [3]"
    )
    parser.add_argument('--filter_Height', type=int, default=2,
        help="filter duration. As ECG is one-dimension signal, height should always be [1]"
    )
    parser.add_argument('--filter_Depth', type=int, default=1,
        help="filter channel. [1]")
    # 3. resnet Input Hyper_parameters
    parser.add_argument('--inputECG_Duration', type=int, default=10,
        help="input ECG Duration [10]"
    )
    parser.add_argument('--inputECG_Height', type=int, default=10,
        help="input ECG duration. As ECG is one-dimension signal, height should always be [1]"
    )
    parser.add_argument('--inputECG_Depth', type=int, default=1,
        help = "input ECG Channel. [1]"
    )
    # 4. data related
    parser.add_argument('--load_data_to_mem', type=bool, default=True,
        help="If [True], we load the data into memory at one time"
    )
    parser.add_argument('--num_data', type=int, default=5000,
        help="Total Number of datas [5000]")
    parser.add_argument('--num_labels', type=int, default=30,
        help="Number of labels [30]"
    )

    return parser.parse_args()




