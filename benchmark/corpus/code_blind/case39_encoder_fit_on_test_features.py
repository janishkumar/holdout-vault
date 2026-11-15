# expect: HV-COD-002
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("data/churn/test_features.csv")
enc = OneHotEncoder(handle_unknown="ignore")
enc.fit(test[["plan", "region"]])
clf = LogisticRegression().fit(enc.transform(train[["plan", "region"]]), train["churn"])
test["churn"] = clf.predict(enc.transform(test[["plan", "region"]]))
test[["customer_id", "churn"]].to_csv("predictions.csv", index=False)
