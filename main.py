#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Program Access
  Created: 09/06/17
  Updated: 09/13/17 --- conv1d
"""
from Hyper_parameters import *
from ECG_Classifier import *

def main(args):
    ecg = ECG_Classifier(args)

    if args.train_flag == True:
        print("Model Construction / Training ...")
        ecg.train()

    if args.test_flag == True:
        print("Model Testing ...")
        ecg.test(args.ckpt_path)

if __name__ == "__main__":
    parser = arg_parser()
    main(parser)
    print('All Finished')