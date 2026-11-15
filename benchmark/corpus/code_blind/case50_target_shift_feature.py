# expect: HV-COD-003
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("data/churn/train.csv").sort_values("signup_date")
y = df["churn"]
X = df[["x1", "x2", "x3", "tenure"]].copy()
X["prev_outcome"] = y.rolling(3).mean().fillna(0)
RandomForestClassifier().fit(X, y)
