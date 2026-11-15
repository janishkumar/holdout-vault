# expect: HV-COD-003
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/churn/train.csv")[["x1", "x2", "x3", "tenure", "churn"]]
A = df.to_numpy(dtype=float)
y = A[:, 4]
n = 800
X = A[:, :]
clf = LogisticRegression(max_iter=300).fit(X[:n], y[:n])
print(clf.score(X[n:], y[n:]))
