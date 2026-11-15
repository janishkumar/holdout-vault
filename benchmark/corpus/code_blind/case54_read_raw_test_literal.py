# expect: HV-COD-004
import pandas as pd
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("raw/test.csv")
cols = ["x1", "x2", "x3"]
clf = LogisticRegression().fit(train[cols], train["churn"])
print("test accuracy", clf.score(test[cols], test["churn"]))
