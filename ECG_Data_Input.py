#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Auto ECG Classifier
  Created: 09/07/17
"""
import numpy as np
import pickle
import sys

# TODO: define this function!!
def load_all_train_data_to_memory():
    '''
    Load all data into memory

    :return data, labels
    '''
    data = None
    labels = None
    return data, labels

# TODO: load all teset data
def load_all_test_data_to_memory():
    '''
    Load all test data into memory

    :return data, labels
    '''
    data = None
    labels = None
    return data, labels



def prepare_train_data(offset, batch_size, total_num_data, all_data=None, all_labels=None):
    '''
    data: 3D np.array [data_num, ECG_duration, ECG_Channel]
    label: 1D np.array [data_num]


    :param offset: offset we use to choose the batches
    :param batch_size:  the batch_size we given
    :param total_num_data: total data number
    :param all_data: all the data in memory
    :param all_labels: all the all_data's labels in memory
    '''
    data = []
    labels = []
    if all_data is None and all_labels is None:
        '''We have to find a way to load data in here'''
        sys.exit('We have to find a way to load data in here')
    elif all_data is not None and all_labels is not None:
        if offset + batch_size >= total_num_data:
            data = all_data[-batch_size:-1, ...]
            labels = all_labels[-batch_size:-1]
        else:
            currentIdx = offset*batch_size
            data = all_data[currentIdx:currentIdx+batch_size, ...]
            labels = all_labels[currentIdx:currentIdx+batch_size]
    else:
        sys.exit('data and labels are not loaded correctly!!!')

    assert data.shape[0] == len(labels)
    return data,labels