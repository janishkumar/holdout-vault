# expect: HV-COD-003, HV-COD-004
import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("data/churn/train.csv")
df["churn_x1"] = df["churn"] * df["x1"]
test = pd.read_csv("raw/test.csv")
test["churn_x1"] = test["churn"] * test["x1"]
cols = ["x1", "churn_x1"]
print(LogisticRegression().fit(df[cols], df["churn"]).score(test[cols], test["churn"]))
