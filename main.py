#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Program Access
  Created: 09/06/17
"""
from Hyper_parameters import *
from ECG_Classifier import *

def main(args):
    ecg = ECG_Classifier(args)
    print(args.train_flag)
    print(args.test_flag)

    if args.train_flag == True:
        ecg.train()

    if args.test_flag == True:
        ecg.test(args.ckpt_path)

if __name__ == "__main__":
    parser = arg_parser()
    main(parser)
    print('All Finished')