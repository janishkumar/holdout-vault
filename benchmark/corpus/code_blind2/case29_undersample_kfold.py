# expect: HV-COD-005
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/churn/train.csv")
X, y = RandomUnderSampler(random_state=0).fit_resample(df[["x1", "x2", "x3", "tenure"]].to_numpy(), df["churn"].to_numpy())
acc = []
for a, b in KFold(5).split(X):
    acc.append(DecisionTreeClassifier(max_depth=4).fit(X[a], y[a]).score(X[b], y[b]))
print(np.mean(acc))
