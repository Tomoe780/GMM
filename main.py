import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from GMM import GMM
from visualization import visualize_clusters


data = pd.read_csv(r"./dataset/adult.csv")
X = data[['age', 'fnlwgt']].values

# 取一部分数据点
num_samples = 2000
random_indices = np.random.choice(X.shape[0], num_samples, replace=False)
X = X[random_indices, :]

# 缩放到 [0, 1] 范围
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# 设置聚类数量
K = 5

print("-----------------------------------------")
print("sklearn库的GMM：")
gmm1 = GaussianMixture(K)
gmm1.fit(X)
labels1 = gmm1.predict(X)
silhouette_score_1 = silhouette_score(X, labels1)
print(f"silhouette_score_1: {silhouette_score_1}")
print("-----------------------------------------")

print("代码复现的GMM：")
gmm2 = GMM(K)
gmm2.fit(X)
labels2 = gmm2.predict(X)
silhouette_score_2 = silhouette_score(X, labels2)
print(f"silhouette_score_2: {silhouette_score_2}")
print("-----------------------------------------")
visualize_clusters(X, labels1, labels2)
