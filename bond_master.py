#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:18:29 2018

@author: k26609
"""

import numpy as np
import string
import random 
from random import randint 
import matplotlib.pyplot as plt
import pandas as pd 

string.ascii_letters
#random.choice(string.ascii_letters) 
def generateBondIssuer(nameLen):
    return "".join([random.choice(string.ascii_letters) for i in range(nameLen)])

numIssuers = 3500
issuers = [generateBondIssuer(randint(4,7)) for i in range(numIssuers)]

plt.hist(np.random.chisquare(15, numIssuers))

issuer_bond_dict = dict() 
totalNumBonds = 0 
for issuer, numBond in zip(issuers, np.random.chisquare(15, numIssuers)): 
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

totalTransactions = 5*10**6
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

#df_trsctn_timeSorted = df_trsctn_timeSorted.set_index(['BondName'])
#df_trsctn_timeSorted.head()
#def getLastTransactionTime(strBondName, floatTime):
#    return df_trsctn_timeSorted[(df_trsctn_timeSorted['BondName'] == strBondName)
#                            & (df_trsctn_timeSorted['Time'] < floatTime)
#                        ]['Time'].max()
#
#df_trsctn_timeSorted['LastTime'] = df_trsctn_timeSorted.apply(lambda row: getLastTransactionTime(row['BondName'], row['Time']), axis = 1)
#getLastTransactionTime('wGhy1', 0.000003)

df_trsctn_timeSorted['LastTime'] = df_trsctn_timeSorted.groupby('BondName').Time.shift(1).fillna(float(0))
#df_trsctn_timeSorted[df_trsctn_timeSorted['BondName'] == 'wGhy1']
df_trsctn_timeSorted.head()


# evenly chunk the huge 
issuer_distribution = df_trsctn_timeSorted.groupby(['issuer']).size().reset_index(name='counts').fillna(0)
sum(list(issuer_distribution['counts'])) == totalTransactions # True
plt.hist(issuer_distribution) 

threads = 10 
approx_size = int(round(len(df_trsctn_timeSorted) / threads )) 
issuers_chunk = [ [] for i in range(threads)] 
issuers_chunk_count = [ 0 for i in range(threads)] 
part_number = 0 
for issuer in issuers: 
#    print("constructing chunk: ", part_number + 1)
    issuers_chunk_count[part_number] += int(issuer_distribution[issuer_distribution['issuer'] == issuer]['counts']) 
    issuers_chunk[part_number].append(issuer) 
    if issuers_chunk_count[part_number] > approx_size: 
        part_number += 1 

for i in range(len(issuers_chunk)):    
    df = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'].isin(issuers_chunk[i])].copy(deep=True)
    fileName = r'./df_trsctn_timeSorted_part_'+str(i+1)+'.pkl'
    df.to_pickle(fileName) 

# to this point, you will have 10 (threads) pickle files in your current folder, each has about the same size/length
# next: open 10 terminals/notebook to run "python bond_mapper.py df_trsctn_timeSorted_part_X.pkl" 10 times, X=0,1,...9





""" validation chunks : 
input_argus = [ 'df_trsctn_timeSorted_part_1.pkl'
               ,'df_trsctn_timeSorted_part_2.pkl'
               ,'df_trsctn_timeSorted_part_3.pkl'
               ,'df_trsctn_timeSorted_part_4.pkl'
               ,'df_trsctn_timeSorted_part_5.pkl'
               ,'df_trsctn_timeSorted_part_6.pkl'
               ,'df_trsctn_timeSorted_part_7.pkl'
               ,'df_trsctn_timeSorted_part_8.pkl'
               ,'df_trsctn_timeSorted_part_9.pkl'
               ,'df_trsctn_timeSorted_part_10.pkl'
               ]

print("Input df_pickle files: ", input_argus)
df_input_list = [] 
for input_argu in input_argus:
    dataframe_name = r'./' + str(input_argu)
    df_input_list.append(pd.read_pickle(dataframe_name))
    
finalResults = pd.concat(df_input_list)
finalResults = finalResults.sort_values('Time').reset_index(drop=True)

finalResults.equals(df_trsctn_timeSorted)


len(set(finalResults['issuer']))
len(set(df_trsctn_timeSorted['issuer'])) 
"""






"""

df_train = df_trsctn_timeSorted[df_trsctn_timeSorted['Time'] > 0.5].reset_index(drop=True).copy(deep=True)
df_train.head() 
len(df_train)
#df_train2 = df_trsctn_timeSorted[df_trsctn_timeSorted['Time'] > 1].sort_values('Time').reset_index(drop=True).copy(deep=True)
#df_train.equals(df_train2) 
#for i, j in zip(df_train.index, df_train2.index):
#    if df_train.iloc[i]['BondName'] != df_train2.iloc[j]['BondName'] or df_train.iloc[i]['Price'] != df_train2.iloc[j]['Price'] or df_train.iloc[i]['Time'] != df_train2.iloc[j]['Time']:
#        print(i)

########## method 1
n = 0 
df_train['targetBondList1'] = " "
for targerIssuer in issuers: 
    n += 1 
    print("Currently processing n = {0}, target issuer: {1}".format(n, targerIssuer)) 
    df_targer_issuer = None 
    df_targer_issuer = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'] == targerIssuer][['BondName','Time']].copy(deep=True)
    for index, trac in df_train[df_train['issuer'] == targerIssuer].iterrows():
        df = df_targer_issuer[  (df_targer_issuer['Time'] >= trac['LastTime'])
                                & (df_targer_issuer['Time'] < trac['Time'])
                                & (df_targer_issuer['BondName'] != trac['BondName'])
                                ].groupby(['BondName']).tail(1)
        df_train.at[index, 'targetBondList1'] = df.values.tolist()
        
"""




"""
########## method 2
m = 0 
df_train['targetBondList2'] = " "
def getBondbySameIssuer(df, strBondName, floatTime, floatLastTime):
    #print("Currently df's length {0}".format(len(df_targer_issuer))) 
    return df[ (df['Time'] >= floatLastTime)
                 & (df['Time'] < floatTime)
                 & (df['BondName'] != strBondName)
                 ].groupby(['BondName']).tail(1).values.tolist()

for targerIssuer in issuers: 
    m += 1 
    print("Currently processing m = {0}, target issuer: {1}".format(m, targerIssuer)) 
#    global df_targer_issuer 
    df_targer_issuer = None
    df_targer_issuer = df_trsctn_timeSorted[df_trsctn_timeSorted['issuer'] == targerIssuer][['BondName','Time']].copy(deep=True)
    df_train[df_train['issuer'] == targerIssuer]['targetBondList'] = df_train.apply(lambda row: getBondbySameIssuer(df_targer_issuer, row['BondName'], row['Time'], row['LastTime']), axis = 1)

#   BondName      Time  Price   issuer
#0  GTMlIfW1  4.999993     44  GTMlIfW
#1    TkddJ2  5.000000     28    TkddJ
#a = getBondbySameIssuer('GTMlIfW1', 4.999993, 'GTMlIfW').values.flatten().tolist()
#len(a)
#getBondbySameIssuer('TkddJ2', 5.000000, 'TkddJ').values.tolist()

df_train.iloc[0]['targetBondList1']
df_train.iloc[0]['targetBondList2']


#df_train.head()
#df_train.drop(['new1'], axis=1, inplace=True)

"""

#plt.hist(df_train.groupby(['issuer']).count()) 
#df_train.to_pickle('./df_train.pkl') 
#new_df = pd.read_pickle('./df_train.pkl') 
#
#len(df_train)
#len(new_df)

#new_df = pd.read_pickle('./df_trsctn_timeSorted_part_1.pkl') 
#a = set(new_df['issuer'])
#len(a) 