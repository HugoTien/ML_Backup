###################################
###      預處理                 ###
###################################

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values #不可以 [:, 1]，因為包含自變數的格式要矩陣
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
    #此次不做(資料太少)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
    #這次一樣不需要，原因如之前。
    
# Fitting Linear Regression to the dataset
    # 創建兩個(只是拿來比較)
    # Linear
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
    
    # Poly
from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 4)
        # degree = 最高次限制
        # degree 2, 3, 4...
    # 1. 用已有的自變量擬合poly
    # 2. 用擬合好的對向轉化自變量
X_poly = poly_reg.fit_transform(X)

# Fitting Polynomial Regression to the dataset
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,  y)

# Visualising the linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), Color = 'blue')
plt.title("Truth or Bluff (Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue') #or 直接用X_poly(方便未來修改)
plt.title("Truth or Bluff (Polynomal Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Let line smooth (縮小間距)
X_grid = np.arange(min(X), max(X), 0.1)
    # start = 開始, stop = 結束, step = 間距
    # 給平均分佈直
X_grid = X_grid.reshape(len(X_grid), 1)
    # 轉化成矩陣
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') #or 直接用X_poly(方便未來修改)
plt.title("Truth or Bluff (Polynomal Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")


# Predicting a new result with Linear Regression
lin_reg.predict(6.5)
# Predicting a new result with Polynomial
lin_reg_2.predict(poly_reg.fit_transform(6.5))