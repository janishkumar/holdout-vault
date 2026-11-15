# expect: HV-COD-003
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/churn/train.csv")
rate = df.groupby("region")["churn"].mean()
df["region_rate"] = df["region"].map(rate)
X = df[["x1", "x2", "tenure", "region_rate"]]
clf = LogisticRegression().fit(X, df["churn"])
print(clf.coef_)
