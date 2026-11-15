# expect: none
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/churn/train.csv")
X = df[["x1", "x2", "x3", "tenure"]].to_numpy()
y = df["churn"].to_numpy()
make = partial(StandardScaler)
scores = []
for tri, vai in KFold(5, shuffle=True, random_state=0).split(X):
    s = make()
    m = LogisticRegression(max_iter=300).fit(s.fit_transform(X[tri]), y[tri])
    scores.append(m.score(s.transform(X[vai]), y[vai]))
print(np.mean(scores))
