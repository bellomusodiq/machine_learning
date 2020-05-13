# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 00:15:26 2019

@author: PC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_set = pd.read_csv('./datasets/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv').iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

X_train = training_set[0:1257]
y_train = training_set[1:1258]

X_train = np.reshape(X_train, (1257,1,1))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

regressor = Sequential()
regressor.add(LSTM(4, activation='sigmoid', input_shape=(None, 1)))

regressor.add(Dense(1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(verbose=1, monitor='loss', patience=5)

regressor.fit(X_train, y_train, batch_size=32, epochs=200)

test_set = pd.read_csv('./datasets/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv').iloc[:, 1:2].values

inputs = test_set
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(np.arange(1,21).reshape(20,1), test_set, color='red')
plt.plot(np.arange(2,22).reshape(20,1), predicted_stock_price,  color='blue')
plt.show()