#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:08:02 2018

@author: k26609
"""

import numpy as np
import string
import random 
from random import randint 
import matplotlib.pyplot as plt
import pandas as pd 

# python bond_mapper.py df_trsctn_timeSorted_part_1.pkl
import sys 

input_argu = sys.argv[1] 
dataframe_name = r'./' + str(input_argu)


print('input file name: ' + dataframe_name)
df_trsctn_timeSorted = pd.read_pickle(dataframe_name) 
#dataframe_name = r'./df_trsctn_timeSorted_part_1.pkl'
#df_trsctn_timeSorted = pd.read_pickle('./df_trsctn_timeSorted_part_1.pkl')

issuers_sublist = list(set(df_trsctn_timeSorted['issuer'])) 
df_trsctn_timeSorted = df_trsctn_timeSorted.sort_values('Time').reset_index(drop=True)
df_train = df_trsctn_timeSorted[df_trsctn_timeSorted['Time'] > 0.5].reset_index(drop=True).copy(deep=True)
#df_train = df_trsctn_timeSorted.copy(deep=True)
df_train.head() 
len(df_train)

########## method 1
n = 0 
df_train['targetBondList1'] = "if you see this, something is wrong"
for targerIssuer in issuers_sublist: 
    n += 1 
    print("Currently processing file {0}, n = {1}, target issuer: {2}".format(dataframe_name.replace('.pkl',''), n, targerIssuer)) 
    df_targer_issuer = None 
    df_targer_issuer = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'] == targerIssuer][['BondName','Time']].copy(deep=True)
    for index, trac in df_train[df_train['issuer'] == targerIssuer].iterrows():
        df = df_targer_issuer[  (df_targer_issuer['Time'] >= trac['LastTime'])
                                & (df_targer_issuer['Time'] < trac['Time'])
                                & (df_targer_issuer['BondName'] != trac['BondName'])
                                ].groupby(['BondName']).tail(1)
        df_train.at[index, 'targetBondList1'] = df.values.tolist()
#        print("current index is: ", index)
#       print(df_train.at[index, 'targetBondList1'])


fileName = dataframe_name.replace('.pkl','') + '_output.pkl'
df_train.to_pickle(fileName) 
print('---------------------------------> mission accomplished: output file: ', fileName)


#df_train.iloc[916]['targetBondList1']
#df_train[df_train['issuer'] == 'AaQnpjk'] 
#for index, trac in df_train[df_train['issuer'] == 'AaQnpjk'].iterrows():
#    print(index) 

