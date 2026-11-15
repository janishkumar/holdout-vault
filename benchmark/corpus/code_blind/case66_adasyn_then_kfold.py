# expect: HV-COD-005
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from imblearn.over_sampling import ADASYN
df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

Xa, ya = ADASYN(random_state=0).fit_resample(X[NUM].fillna(0), y)
Xa, ya = np.asarray(Xa), np.asarray(ya)
scores = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, ya):
    scores.append(LogisticRegression(max_iter=500).fit(Xa[tr], ya[tr]).score(Xa[va], ya[va]))
print(np.mean(scores))
