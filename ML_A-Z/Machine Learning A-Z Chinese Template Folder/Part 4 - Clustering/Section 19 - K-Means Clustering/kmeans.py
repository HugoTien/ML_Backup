# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##########################
# Importing the dataset
##########################
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

    # 集群問題不用自變應變量
    # 不用特徵縮放
    # 不用分訓練測試集
    
###############################################################
# Using the elbow method to find the optimal number of clusters
# 確定要分幾群
###############################################################
from sklearn.cluster import KMeans
wcss = []
    # 建立個向量(不同組數的組間距離)
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    #利用FOR回圈，分別計算不同組數的組間距離
        # n_clusters = 集群數
        # MAX_ITER = 每次計算的最大循環數(?)
        # n_init = 對多少組不同的中心值進行運算
        # init = 如何選取初始值(用random最簡單，但不好)
        # kmeans.inertia_ = kmans組間距離和
plt.plot(range(1,11), wcss)
    # 橫軸1~10
    # 縱軸wcss
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
    # 如圖所看，5群最好

############################################
# Applying the k-means to the mall dataset
############################################
kmeans = KMeans(n_clusters = 5, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
    # 用此來預測
y_kmeans = kmeans.fit_predict(X)
    # 把結果都放進y_kmeans
    # 擬合後預測兩步驟一次搞定 結果為分到哪一群

################################      
# Visualizing the clusters
################################
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful') #Label可先123，再改
    # X [y_keman = 0 (數據=0第零組), 且第零欄(也只有這欄)]
    # size 100
    # color red
    # label cluster
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')

plt.scatter(kmeans.cluster_centers_[:, 0],  kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    # Center: 群組中心點

plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend() # 代碼
plt.show()

