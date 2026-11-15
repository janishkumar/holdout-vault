# expect: HV-COD-003
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("data/churn/train.csv")
df["churn_x2"] = df["churn"] * df["x2"]
feats = ["x1", "x2", "x3", "tenure", "churn_x2"]
rf = RandomForestClassifier(random_state=0)
rf.fit(df[feats], df["churn"])
print(rf.feature_importances_)
