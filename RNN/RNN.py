#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 23:04:06 2018

@author: k26609
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
 
dataset_train = pd.read_csv(r'/Users/k26609/Documents/GitHub/DeepLearning/RNN/GOOGL.csv') 

training_set = dataset_train.iloc[:1258, 1:2].values 

sc = MinMaxScaler(feature_range = (0 ,1)) 
training_set_scaled = sc.fit_transform(training_set) 

X_train = [] 
y_train = [] 
for i in range(60, 1258): 
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
# this is to use the past 60 days' open prices to predict tomorrow's open price    
# X[0:59] => Y[0] = X[60]  
X_train, y_train = np.array(X_train), np.array(y_train) 

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') 

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 

# testing

#dataset_test = pd.read_csv(r'') 
dataset_test = dataset_train.iloc[1258+60:, :]
real_stock_price = dataset_test.iloc[:, 1:2].values 

#dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values 

inputs = dataset_test.iloc[:, 1:2].values 
inputs_scaled = sc.transform(inputs)

X_test = [] 
real_stock_price = [] 
for i in range(60,80):
    X_test.append(inputs_scaled[i-60:i, 0])
    real_stock_price.append(inputs[i])

X_test = np.array(X_test)
real_stock_price = np.array(real_stock_price)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()









