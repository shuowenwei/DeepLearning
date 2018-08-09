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



import numpy as np
import string
import random 
from random import randint 
import matplotlib.pyplot as plt
import pandas as pd 

# python bond_mapper.py df_trsctn_timeSorted_part_1.pkl
import sys 

input_argus = [] 
for fileName in sys.argv[1:]:
    input_argus.append(fileName)

df_input_list = [] 
for input_argu in input_argus:
    dataframe_name = r'./' + str(input_argu)
    df_input_list.append(pd.read_pickle(dataframe_name))
    
finalResults = pd.concat(df_input_list)
finalResults = finalResults.sort_values('Time').reset_index(drop=True)

finalResults.to_pickle('./finalResults')