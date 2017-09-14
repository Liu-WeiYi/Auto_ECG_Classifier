#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Define hyper_parameters
  Created:  09/06/17
  Updated:  09/13/17 --- conv1d
"""
import argparse
"""
IMPORTANT Hyper-Parameters
ECG Signal Definition
************************************************************************
1. Filter Duration (Width):     100ms
2. inputECG Duration(Width):    3s = 3000ms
3. num_data (Total Samples):    10000
4. num_labels (Total Labels):   30
************************************************************************
Deep Model Related
    --- Residual Block Conv Layer: BN->ReLu[->dropout]->conv
    --- Each Residual Block has two residual block conv layers
************************************************************************
1. original_res:                True original resnet with 5 conv blocks and 32 layers
2. init_f_channels:             32   the first filter output channels number 32
3. residual_block_num:          5    Total residual block (BN->Relu->[dropout->]Conv) number
4. keep_prob:                   0.5  Keep probability in Dropout Layer

************************************************************************
"""

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
    parser.add_argument('--train_flag', type=bool, default=True,
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
    parser.add_argument('--lr_decay_rate', type=float, default=0.96,
        help="learning rate decay value [0.1]"
    )
    parser.add_argument('--lr_decay_step', type=int, default=10000,
        help="learning rate decay step [5000]"
    )
    parser.add_argument('--regularization_weight_decay', type=float, default=0.0002,
        help="weight decay scale for l2 regularization [0.0002]"
    )
    # 2. resnet Related Hyper_parameters
    parser.add_argument('--residual_block_num', type=int, default=5,
        help="residual block num. [5]"
    )
    parser.add_argument('--filter_Duration', type=int, default=100,
        help="filter duration [100]"
    )
    parser.add_argument('--filter_Depth', type=int, default=1,
        help="filter channel. [1]"
    )
    parser.add_argument('--init_f_channels', type=int, default=32,
        help="initial filter output channels number"
    )
    parser.add_argument('--keep_prob', type=float, default=0.5,
        help="Keep probability in dropout layer"
    )
    # 3. resnet Input Hyper_parameters
    parser.add_argument('--inputECG_Duration', type=int, default=3000,
        help="input ECG Duration [3000]"
    )
    parser.add_argument('--inputECG_Depth', type=int, default=1,
        help = "input ECG Channel. [1]"
    )
    # 4. data related
    parser.add_argument('--load_all_data_to_memory', type=bool, default=True,
        help="If [True], we load all training data into memory in preprocessing step"
    )
    parser.add_argument('--num_data', type=int, default=5000,
        help="Total Number of data [5000]")
    parser.add_argument('--num_labels', type=int, default=30,
        help="Number of labels [30]"
    )

    return parser.parse_args()




