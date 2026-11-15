# expect: HV-COD-001
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

sc = StandardScaler().fit(X[NUM])
Xs = sc.transform(X[NUM])
res = []
for tr, va in KFold(5, shuffle=True, random_state=0).split(Xs):
    m = LogisticRegression(max_iter=500).fit(Xs[tr], y.iloc[tr])
    res.append(m.score(Xs[va], y.iloc[va]))
print(np.mean(res))
