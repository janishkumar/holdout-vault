import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils import balanced_xy
X, y = balanced_xy("data/churn/train.csv")
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)
print(LogisticRegression(max_iter=500).fit(X_tr, y_tr).score(X_va, y_va))
