import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 1) Prepare data
# X_train: only NORMAL data
# X_test: mix of normal + anomalies
rng = np.random.RandomState(42)
X_normal = 0.3 * rng.randn(1000, 2)  + np.array([0, 0])
X_anom   = rng.uniform(low=-4, high=4, size=(40, 2))  # anomalies
X_train  = X_normal[:800]
X_test   = np.vstack([X_normal[800:], X_anom])

# 2) Scale features 
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3) Train One-Class SVM
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
ocsvm.fit(X_train_s)

# 4) Predict on new data
scores = ocsvm.decision_function(X_test_s).ravel()
labels = ocsvm.predict(X_test_s)     

# Rank anomalies by score ascending 
anom_rank = np.argsort(scores)
top_k = 10
top_anomalies_idx = anom_rank[:top_k]
print("\n")
print("\n")
print("01-Support Vector Machine for Novelty Detection")
print("Top anomalies:", top_anomalies_idx)
print("\n")