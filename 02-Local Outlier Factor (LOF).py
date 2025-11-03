import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# 1) Prepare data
rng = np.random.RandomState(0)
X_cluster = 0.2 * rng.randn(800, 2) + np.array([1.0, 1.0])
X_sparse  = rng.uniform(low=-3.5, high=3.5, size=(40, 2))  # anomalies
X = np.vstack([X_cluster, X_sparse])

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)

# 2) Fit LOF
lof = LocalOutlierFactor(n_neighbors=35, contamination=0.05, novelty=False)
labels = lof.fit_predict(X_s) 

# LOF negative_outlier_factor_: lower = more anomalous
lof_scores = -lof.negative_outlier_factor_  
rank = np.argsort(-lof_scores)  

top_k = 10
print("\n")
print("\n")
print("02-Local Outlier Factor (LOF)")
print("Top-10 anomalies:", rank[:top_k])
print("\n")
