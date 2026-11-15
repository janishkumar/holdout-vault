# expect: none
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier

arr = pd.read_csv("data/churn/train.csv")[["x1", "x2", "x3", "tenure", "churn"]].to_numpy(dtype=float)
rng = np.random.default_rng(1)
rng.shuffle(arr)
n = int(len(arr) * 0.7)
tr, va = np.split(arr, [n])
X, y = tr[:, :-1], tr[:, -1]
sc = MinMaxScaler()
X = sc.fit_transform(X)
clf = RidgeClassifier().fit(X, y)
test_df = va
print((clf.predict(sc.transform(test_df[:, :-1])) == test_df[:, -1]).mean())
