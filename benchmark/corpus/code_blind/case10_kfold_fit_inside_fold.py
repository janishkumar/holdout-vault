# expect: none
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

kf = KFold(n_splits=5, shuffle=True, random_state=0)
aucs = []
for tr_idx, va_idx in kf.split(X):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X.iloc[tr_idx][NUM])
    Xva = sc.transform(X.iloc[va_idx][NUM])
    m = LogisticRegression(max_iter=500).fit(Xtr, y.iloc[tr_idx])
    aucs.append(m.score(Xva, y.iloc[va_idx]))
print(np.mean(aucs))
