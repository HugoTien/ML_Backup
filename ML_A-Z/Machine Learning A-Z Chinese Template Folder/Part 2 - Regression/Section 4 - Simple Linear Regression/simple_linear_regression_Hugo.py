# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 01:59:55 2018

@author: Hao-Ping
"""

# Simple Linear Regression


###### Data Preprocessing Template #############
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
#################################################
# python 簡單線性回歸已經自動包含特徵縮放，不用再做
#################################################

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #這次不需要給任何參數值，直接預設
regressor.fit(X_train, y_train)
    # parameter:
        # X = 自變量矩陣
        # y = 因變量
    # 此回歸器如此設置完畢，之後即可使用來測試測試集之數據

# Predicting the Test set results
y_pred = regressor.predict(X_test)
    # parameter:
        # x = 自變量矩陣

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red') # 點圖
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # 迴歸線
plt.title('Salary vs Experience (training set)') # 標題
plt.xlabel('Years of Experience') # x軸名稱
plt.ylabel('Salary') # y軸名稱
plt.show

# Visualising the Testing set result
plt.scatter(X_test, y_test, color = 'red') # 點圖
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # 迴歸線 # 記得仍然是Xtrain(因為是跟用X_train訓練的回歸進行比較)
plt.title('Salary vs Experience (Test set)') # 標題
plt.xlabel('Years of Experience') # x軸名稱
plt.ylabel('Salary') # y軸名稱
plt.show


