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


# python bond_reducer.py df_trsctn_timeSorted_part_1_output.pkl df_trsctn_timeSorted_part_2_output.pkl
import sys 

input_argus = [] 
for fileName in sys.argv[1:]:
    input_argus.append(fileName)
"""
input_argus = [ 'df_trsctn_timeSorted_part_1_output.pkl'
               ,'df_trsctn_timeSorted_part_2_output.pkl'
#               ,'df_trsctn_timeSorted_part_3_output.pkl'
#               ,'df_trsctn_timeSorted_part_4_output.pkl'
#               ,'df_trsctn_timeSorted_part_5_output.pkl'
#               ,'df_trsctn_timeSorted_part_6_output.pkl'
#               ,'df_trsctn_timeSorted_part_7_output.pkl'
#               ,'df_trsctn_timeSorted_part_8_output.pkl'
#               ,'df_trsctn_timeSorted_part_9_output.pkl'
#               ,'df_trsctn_timeSorted_part_10_output.pkl'
               ]
"""

print("Input df_pickle files: ", input_argus)
df_input_list = [] 
for input_argu in input_argus:
    dataframe_name = r'./' + str(input_argu)
    df_input_list.append(pd.read_pickle(dataframe_name))
    
finalResults = pd.concat(df_input_list)
finalResults = finalResults.sort_values('Time').reset_index(drop=True)

finalResults.to_pickle('./finalResults')

temp = pd.read_pickle('./df_trsctn_timeSorted_part_1_output.pkl')



