#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:46:15 2018

@author: k26609
"""

import numpy as np
import string
import random 
from random import randint 
import matplotlib.pyplot as plt
import pandas as pd 

string.ascii_letters
def generateBondIssuer(nameLen):
    return "".join([random.choice(string.ascii_letters) for i in range(nameLen)])

numIssuers = 150
issuers = [generateBondIssuer(randint(4,7)) for i in range(numIssuers)]

plt.hist(np.random.chisquare(100000, numIssuers))

issuer_bond_dict = dict() 
totalNumBonds = 0 
for issuer, numBond in zip(issuers, np.random.chisquare(100000, numIssuers)): 
    issuer_bond_dict[issuer] = [issuer+str(i) for i in range(int(numBond))] 
    totalNumBonds += int(numBond) 
print("Total number of bonds: ", totalNumBonds) 

def getABond():
    issuer = random.choice(issuers) 
    return random.choice(issuer_bond_dict[issuer])

def getATime():
    return random.uniform(0,5) 

def getAPrice():
    return random.randint(10,50) 

totalTransactions = 10**6
print("Total number of transactions: ", totalTransactions) 
transactionsList = [] 
for i in range(totalTransactions):
    transactionsList.append([getABond(), getATime(), getAPrice()])


columnNames = ['BondName', 'Time', 'Price']
df_trsctn = pd.DataFrame(transactionsList, columns=columnNames, index=range(totalTransactions))
bond_issuedby_dict = dict()
for issuer, bondList in issuer_bond_dict.items():
    for bond in bondList:
        bond_issuedby_dict[bond] = issuer
        
df_trsctn['issuer'] = df_trsctn['BondName'].apply(lambda bondname: bond_issuedby_dict[bondname])

df_trsctn_timeSorted = df_trsctn.sort_values('Time').reset_index(drop=True).copy(deep=True)
df_trsctn_timeSorted.head()

# get last transaction time for the same bond (gourp by "BondName")
df_trsctn_timeSorted['LastTime'] = df_trsctn_timeSorted.groupby('BondName').Time.shift(1).fillna(float(0))
#df_trsctn_timeSorted[df_trsctn_timeSorted['BondName'] == 'wGhy1']
df_trsctn_timeSorted.head()


# evenly chunk the huge 
issuer_distribution = df_trsctn_timeSorted.groupby(['issuer']).size().reset_index(name='counts').fillna(0)
sum(list(issuer_distribution['counts'])) == totalTransactions # True
plt.hist(issuer_distribution) 

threads = 400 
approx_size = int(round(len(df_trsctn_timeSorted) / threads )) # size of each chunk
issuers_chunk = [ [] for i in range(threads)] 
issuers_chunk_count = [ 0 for i in range(threads)] 
part_number = 0 
for issuer in issuers: 
#    print("constructing chunk: ", part_number + 1)
    issuers_chunk_count[part_number] += int(issuer_distribution[issuer_distribution['issuer'] == issuer]['counts']) 
    issuers_chunk[part_number].append(issuer) 
    if issuers_chunk_count[part_number] > approx_size: 
        part_number += 1 


import time
time_start = time.time()
import gc
gc.collect() 
leftCol = ['BondName', 'Time', 'Price', 'issuer', 'LastTime']
rightCol = ['BondName', 'Time', 'Price', 'issuer']
results = [] 
def convert2List(df_input):
    return [i for i in df_input.values.tolist()]
for i in range(len(issuers_chunk)):
    df = None
    print("Currently process {0} out of {1}:".format(i, len(issuers_chunk)))
    gc.collect()
    df = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'].isin(issuers_chunk[0])].copy(deep=True)
    r = pd.merge(df[leftCol], df[rightCol], on = 'issuer', how = 'left')
    len(r)
    r = r[(r['Time_x'] > r['Time_y'])
            &(r['LastTime'] < r['Time_y'])
#            &(r['BondName_y'].isin(dict_bond[r['BondName_x']]))
        ].sort_values('Time_y').groupby(['BondName_x','Time_x','issuer','BondName_y']).tail(1).groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer'])['BondName_y','Time_y', 'Price_y'].apply(convert2List).reset_index() 
    len(r)
#    r = r.groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer'])['BondName_y','Time_y', 'Price_y'].apply(convert2List).reset_index() 
    len(r)
    results.append(r)
time_end = time.time()
print(time_end - time_start)


df = None 
df = results[0].copy(deep=True)
df.columns 
df_distribution = df.groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer']).size() .reset_index(name='counts').fillna(0)
len(df_distribution)
sum(list(df_distribution['counts'])) == len(df) 
plt.hist(df_distribution) 

df_convert2List = df.groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer'])['BondName_y','Time_y', 'Price_y'].apply(convert2List).reset_index()


""" draft and experiment 

len(df[df['BondName_x'] == 'HhicQKh10'])
df_distribution[df_distribution['BondName_x'] == 'HhicQKh10'] 

df_distribution.columns 

df_groupby = df.groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer'])['BondName_y','Time_y', 'Price_y']

df_groupby = df.groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer'])['BondName_y','Time_y', 'Price_y'].apply(",".join).reset_index()   

groups = df_groupby.groups 
indices = df_groupby.indices 

df['targetList'] = df[df_groupby.indices]['BondName_y','Time_y', 'Price_y'].to_list() 
df.columns 


a = df_groupby.describe() 
# https://pandas.pydata.org/pandas-docs/stable/groupby.html
def myfun(df_input):
#    temp = [] 
#    for index, data in df_input.iterrows():
#        temp.append([data['BondName_y'], data['Time_y'], data['Price_y'] ])
#    return temp 
    return [i for i in df_input.values.tolist()]
            
a = df.groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer'])['BondName_y','Time_y', 'Price_y'].apply(myfun).reset_index() 

df_groupby_list

df_groupby_list = df.groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer'])['BondName_y','Time_y', 'Price_y'].apply(list) 

df['target_list'] = df.groupby(['BondName_x', 'Time_x', 'Price_x', 'issuer']).apply(list).reset_index(drop=True) 

"""




#import pickle 
#with open(r'/Users/k26609/Documents/GitHub/DeepLearning/results.pkl', 'wb') as output_file: 
#    pickle.dump(results,output_file)






"""
#    df = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'].isin(issuers_chunk[i])].copy(deep=True)
#    fileName = r'./df_trsctn_timeSorted_part_'+str(i+1)+'.pkl'
#    df.to_pickle(fileName) 

df = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'].isin(issuers_chunk[0])].reset_index(drop=True).copy(deep=True)
len(df)
results = pd.merge(df[leftCol], df[rightCol], on = 'issuer', how = 'left')
results.columns
len(results)

results = results[(results['Time_x'] > results['Time_y'])
                        &(results['LastTime_x'] < results['Time_y'])
                    ]
len(results)
results = results[(results['Time_x'] > results['Time_y'])
                        &(results['LastTime_x'] < results['Time_y'])
                    ].sort_values('Time_y').groupby(['BondName_x','Time_x','issuer','BondName_y']).tail(1).reset_index(drop=True)
len(results)







# convert to sql: 
#https://stackoverflow.com/questions/30627968/merge-pandas-dataframes-where-one-value-is-between-two-others

import sqlite3
import pandasql as ps


conn = sqlite3.connect(':memory:')
df_train.to_sql('df_train', conn, index=False)
df_trsctn_timeSorted.to_sql('df_trsctn_timeSorted', conn, index=False)
query = '''
select
    df_train.issuer,
    df_train.Time, 
    df_train.LastTime, 
    df_train.BondName as targetBondName, 
    df_trsctn_timeSorted.BondName as assocBondName,
    df_trsctn_timeSorted.Time as assocTime 
from df_train 
left join df_trsctn_timeSorted
    on df_train.issuer = df_trsctn_timeSorted.issuer 
where df_train.Time > df_trsctn_timeSorted.issuer 
    and df_train.LastTime <= df_trsctn_timeSorted.issuer
    and df_train.BondName != df_trsctn_timeSorted.BondName 

'''
df = pd.read_sql_query(query, conn)

"""

