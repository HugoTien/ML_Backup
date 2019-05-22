# Multiple Linear Regression
################################################
###### Data Preprocessing Template #############
################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # 此時有string，所以只能在console看
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# 此例子自變量要encoding, 因變量不用
# 提醒: 注意虛擬變量陷阱，應該要刪掉一個column
    # 但我們的python package multiple regression會自動處理
    # 但還是做做看避免未來忘記
X = X[:, 1:]          

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
#################################################
# python 多元線性回歸已經自動包含特徵縮放，不用再做
#################################################

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set results
y_pred = regressor.predict(X_test)

###################################################################
###### Building the optimal model using Backward Elimination ######
###################################################################
# 反向傳播事情準備
import statsmodels.formula.api as sm
    # b0(常數項)(相當於b0 * 1)
    # 建立一個11111，後面再append原矩陣
X_train = np.append(arr = np.ones((40,1)), values = X_train, axis = 1)
    # np.append
        # arr = 要被加入新矩陣的矩陣
        # values = 要加的矩陣
        # axis = 0 (add row), 1 (add column)
    # np.ones
        # shape = 新矩陣大小
# Step 1: Select a significance level to stay in model
    # 先all in參數，之後會被改變
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]

# Step 2: Fit the full model with all possible predictors
    # 反向淘汰時，迴歸器用的就跟之前創的不一樣了，用新的package
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
    # endog = 因變數
    # exog = 自變數
    
# Step 3: Consider the predictor with the highest p-value, go to step 4,otherwise go
    # How to choose : python超強function，summary
regressor_OLS.summary()
    # find x2's p_value is top
# Step 4: Remove the predictor
    # so remove x2 and fit again
X_opt = X_train[:, [0, 1, 3, 4, 5]]   
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 4, 5]]   
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 5]]   
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() 
regressor_OLS.summary()

# 只剩行政開銷(0.07)，是否要保留? 看考量，但這次選0.05就0.05
        # 未來會教判斷要不要保留這個
X_opt = X_train[:, [0, 3]]   
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit() 
regressor_OLS.summary() 
# p_value = 這裡0.00是因為太小，不是0
  
# Step 5: Fit model without this variable





