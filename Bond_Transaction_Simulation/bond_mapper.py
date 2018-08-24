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

# command to run at the terminal: python bond_mapper.py df_trsctn_timeSorted_part_1.pkl
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
import time
time_start = time.time()
n = 0 
df_train['targetBondList1'] = "if you see this, something is wrong"
for targerIssuer in issuers_sublist: 
    n += 1 
    print("Currently processing file {0}, n = {1}, target issuer: {2}".format(dataframe_name.replace('.pkl',''), n, targerIssuer)) 
    df_targer_issuer = None 
    df_targer_issuer = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'] == targerIssuer][['BondName','Time']].copy(deep=True)
    for index, trac in df_train[df_train['issuer'] == targerIssuer].iterrows():
        df_train.at[index, 'targetBondList1'] = df_targer_issuer[ (df_targer_issuer['Time'] > trac['LastTime'])
                                & (df_targer_issuer['Time'] < trac['Time'])
#                                & (df_targer_issuer['BondName'] != trac['BondName'])
                                ].groupby(['BondName']).tail(1).values.tolist()
#        df_train.at[index, 'targetBondList1'] = df.values.tolist()
#        print("current index is: ", index)
#       print(df_train.at[index, 'targetBondList1'])
fileName = dataframe_name.replace('.pkl','') + '_output.pkl'
df_train.to_pickle(fileName) 
time_end = time.time()
print(time_end - time_start)
print('---------------------------------> mission accomplished: output file: ', fileName)


#df_train.iloc[916]['targetBondList1']
#df_train[df_train['issuer'] == 'AaQnpjk'] 
#for index, trac in df_train[df_train['issuer'] == 'AaQnpjk'].iterrows():
#    print(index) 

"""
########## method 2
import time
time_start = time.time()
m = 0 
df_train['targetBondList2'] = " "
def getBondbySameIssuer(strBondName, floatTime, floatLastTime):
    #print("Currently df's length {0}".format(len(df_targer_issuer))) 
    return df_targer_issuer[ (df_targer_issuer['Time'] > floatLastTime)
                 & (df_targer_issuer['Time'] < floatTime)
#                 & (df['BondName'] != strBondName)
                 ].groupby(['BondName']).tail(1).values.tolist()

for targerIssuer in issuers_sublist: 
    m += 1 
    print("Currently processing m = {0}, target issuer: {1}".format(m, targerIssuer)) 
    global df_targer_issuer 
    df_targer_issuer = None
    df_targer_issuer = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'] == targerIssuer][['BondName','Time']].copy(deep=True)
    df_train.loc[df_train['issuer'] == targerIssuer, 'targetBondList2'] = df_train.apply(lambda row: getBondbySameIssuer(row['BondName'], row['Time'], row['LastTime']), axis = 1)
#    df_train[df_train['issuer'] == targerIssuer]['targetBondList2'] = getBondbySameIssuer( 
#                df_train[df_train['issuer'] == targerIssuer]['BondName'], 
#                df_train[df_train['issuer'] == targerIssuer]['Time'], 
#                df_train[df_train['issuer'] == targerIssuer]['LastTime']
#                ) 


time_end = time.time()
print(time_end - time_start)





leftCol = df_train.columns 
rightCol = df_trsctn_timeSorted.columns 

results = pd.merge(df_train[leftCol], df_trsctn_timeSorted[rightCol], on = 'issuer', how = 'left')
results.columns
len(results)
results = results[(results['Time_x'] > results['Time_y'])
                        &(results['LastTime_x'] < results['Time_y'])
#                        &(results['BondName_x'] != results['BondName_y'])
                        ]
len(results_filter)


"""

