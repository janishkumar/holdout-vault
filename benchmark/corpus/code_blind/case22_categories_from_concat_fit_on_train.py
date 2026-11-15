# expect: none
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("data/churn/test_features.csv")
# only used to list the region values that can occur, nothing is fitted on the test rows
regions = sorted(set(pd.concat([train["region"], test["region"]]).dropna().unique()))
enc = OneHotEncoder(categories=[regions], handle_unknown="ignore")
Xtr = enc.fit_transform(train[["region"]])
clf = LogisticRegression().fit(Xtr, train["churn"])
test["churn"] = clf.predict(enc.transform(test[["region"]]))
test[["customer_id", "churn"]].to_csv("predictions.csv", index=False)
