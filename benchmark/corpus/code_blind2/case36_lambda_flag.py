# expect: HV-COD-003
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv")
flag = lambda v: 1 if v else 0
df["flag"] = df["churn"].map(flag)
feats = ["x1", "x2", "tenure", "flag"]
a, b = train_test_split(df, test_size=0.2, random_state=0)
m = CatBoostClassifier(iterations=50, verbose=0).fit(a[feats], a["churn"])
print(m.score(b[feats], b["churn"]))
