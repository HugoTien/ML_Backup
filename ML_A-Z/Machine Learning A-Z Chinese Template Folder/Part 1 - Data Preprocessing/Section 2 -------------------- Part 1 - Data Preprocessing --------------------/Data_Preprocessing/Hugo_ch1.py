# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt #只要matplotlib的subpackage(畫圖用)
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Data.csv")

#python + 機器學習裡面，創造自變量矩陣 & 因變量向量是必要的 !
X = dataset.iloc[:, :-1].values # iloc用來指定行列 #第一個冒號，代表取全部列 #第二個冒號，除最後一個行外都取
Y = dataset.iloc[:, 3].values # 因變量向量