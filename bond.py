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
    issuer_bond_dict[issuer] = [issuer+str(int(numBond)) for i in range(int(numBond))] 
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
pd_transactions = pd.DataFrame(transactionsList, columns=columnNames, index=range(totalTransactions))
pd_transactions.head()

###############################################################################
bond_issuedby_dict = dict()
for issuer, bondList in issuer_bond_dict.items():
    for bond in bondList:
        bond_issuedby_dict[bond] = issuer
        
pd_transactions['issuer'] = pd_transactions['BondName'].apply(lambda bondname: bond_issuedby_dict[bondname])
pd_transactions.head()

pd_transactions[pd_transactions['Time'] < 4.7].groupby(['issuer', 'BondName'])['Price'].max() 


issuer = ''
pd_transactions[pd_transactions['Time'] < 4.7].groupby(['issuer', 'BondName'])['Price'].max()
pd_transactions[pd_transactions['issuer'] = issuer].groupby(['issuer', 'BondName'])['Price'].max()











