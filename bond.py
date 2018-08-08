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

issuers = [generateBondIssuer(randint(4,7)) for i in range(3500)]

plt.hist(np.random.chisquare(15,3500))

issuer_bond_dict = dict() 
totalNumBonds = 0 
for issuer, numBond in zip(issuers, np.random.chisquare(15,3500)): 
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

###############################################################################
"""
df_trsctn[df_trsctn['Time'] < 4.7].groupby(['issuer', 'BondName'])['Price'].max() 

issuer = 'RcAJ'
df_trsctn[df_trsctn['Time'] < 4.7].groupby(['issuer', 'BondName'])['Price'].max()
df_trsctn[(df_trsctn['issuer'] == issuer) & (df_trsctn['Time'] < 4.7) ].sort_values('Time') 

df = df_trsctn[(df_trsctn['issuer'] == issuer)
          & (df_trsctn['Time'] < 4.7)
          & (df_trsctn['BondName'] != 'anything')].groupby(['BondName']).head(1)[['BondName','Price']]
"""
###############################################################################


df_train = df_trsctn_timeSorted[df_trsctn_timeSorted['Time'] > 1].reset_index(drop=True).copy(deep=True)
df_train.head() 
len(df_train)
#df_train2 = df_trsctn_timeSorted[df_trsctn_timeSorted['Time'] > 1].sort_values('Time').reset_index(drop=True).copy(deep=True)
#df_train.equals(df_train2) 
#for i, j in zip(df_train.index, df_train2.index):
#    if df_train.iloc[i]['BondName'] != df_train2.iloc[j]['BondName'] or df_train.iloc[i]['Price'] != df_train2.iloc[j]['Price'] or df_train.iloc[i]['Time'] != df_train2.iloc[j]['Time']:
#        print(i)

########## method 1
n = 0 
df_train['targetBondList2'] = " "
for index, trac in df_train.iterrows():
    n += 1 
    print("currently processing n = ", n)
    df = df_trsctn_timeSorted[(df_trsctn_timeSorted['issuer'] == trac['issuer'])
                                & (df_trsctn_timeSorted['Time'] >= trac['LastTime'])
                                & (df_trsctn_timeSorted['Time'] < trac['Time'])
                                & (df_trsctn_timeSorted['BondName'] != trac['BondName'])
                                ].groupby(['BondName']).tail(1)
    df_train.at[index, 'targetBondList2'] = df.values.tolist()


########## method 2
def getBondbySameIssuer(strBondName, floatTime, strIssuer, floatLastTime):
    return df_trsctn_timeSorted[(df_trsctn_timeSorted['issuer'] == strIssuer)
                                & (df_trsctn_timeSorted['Time'] >= floatLastTime)
                                & (df_trsctn_timeSorted['Time'] < floatTime)
                                & (df_trsctn_timeSorted['BondName'] != strBondName)
                                ].groupby(['BondName']).tail(1).values.tolist()

df_train['targetBondList'] = df_train.apply(lambda row: getBondbySameIssuer(row['BondName'], row['Time'], row['issuer'], row['LastTime']), axis = 1)
#   BondName      Time  Price   issuer
#0  GTMlIfW1  4.999993     44  GTMlIfW
#1    TkddJ2  5.000000     28    TkddJ
#a = getBondbySameIssuer('GTMlIfW1', 4.999993, 'GTMlIfW').values.flatten().tolist()
#len(a)
#getBondbySameIssuer('TkddJ2', 5.000000, 'TkddJ').values.tolist()

df_train.iloc[0]['targetBondList']
df_train.iloc[0]['targetBondList2']




#df_train.head()
#df_train.drop(['new1'], axis=1, inplace=True)

