# expect: HV-COD-003
import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("data/churn/train.csv")
# historical churn rate per region as a feature
df["region_churn_rate"] = df.groupby("region")["churn"].transform("mean")
feats = ["x1", "x2", "tenure", "region_churn_rate"]
clf = LogisticRegression().fit(df[feats], df["churn"])
print(clf.coef_)
