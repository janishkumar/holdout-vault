# expect: none
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = []
for tr, va in skf.split(X[NUM], y):
    Xr, yr = RandomOverSampler(random_state=0).fit_resample(X.iloc[tr][NUM], y.iloc[tr])
    m = LogisticRegression(max_iter=500).fit(Xr, yr)
    scores.append(m.score(X.iloc[va][NUM], y.iloc[va]))
print(np.mean(scores))
