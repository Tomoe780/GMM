import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import pandas as pd
from visualization import visualize_clusters


data = pd.read_csv(r"./dataset/bank.csv")
X = data[['balance', 'duration']].values

# 取一部分数据点
num_samples = 1000
random_indices = np.random.choice(X.shape[0], num_samples, replace=False)
X = X[random_indices, :]

# 缩放到 [0, 1] 范围
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# 设置聚类数量
K = 6

print("-----------------------------------------")
print("sklearn库的GMM：")
gmm = GaussianMixture(K)
gmm.fit(X)
labels1 = gmm.predict(X)
silhouette_score_GMC = silhouette_score(X, labels1)
print(f"silhouette_score_GMC: {silhouette_score_GMC}")
print("-----------------------------------------")

print("代码复现的GMM：")
gmm2 = GaussianMixture(K)
gmm2.fit(X)
labels2 = gmm.predict(X)
silhouette_score_IFGMC = silhouette_score(X, labels2)
print(f"silhouette_score_IFGMC: {silhouette_score_IFGMC}")
print("-----------------------------------------")
visualize_clusters(X, labels1, labels2)
