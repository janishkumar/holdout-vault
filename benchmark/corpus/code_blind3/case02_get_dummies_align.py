# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("data/churn/test_features.csv")
cats = ["plan", "colour", "region"]
y = train.pop("churn")
ids = test["customer_id"]
Xtr = pd.get_dummies(train.drop(columns=["customer_id", "signup_date"]), columns=cats)
Xte = pd.get_dummies(test.drop(columns=["customer_id", "signup_date"]), columns=cats)
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
clf = LogisticRegression(max_iter=500).fit(Xtr, y)
pd.DataFrame({"customer_id": ids, "p": clf.predict_proba(Xte)[:, 1]}).to_csv("out_preds.csv", index=False)
