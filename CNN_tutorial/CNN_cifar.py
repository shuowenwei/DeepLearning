#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:55:20 2018

@author: k26609
"""

import pickle 

CIFAR_DIR = 'cifar-10-batches-py/'
def unpickle(file): 
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict 


    

