# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

num = ["x1", "x2", "x3", "tenure"]
train = pd.read_csv("data/churn/train.csv")
scaler = StandardScaler()
Xs = scaler.fit_transform(train[num])
clf = LogisticRegression().fit(Xs, train["churn"])
test = pd.read_csv("data/churn/test_features.csv")
pred = clf.predict_proba(scaler.transform(test[num]))[:, 1]
pd.DataFrame({"customer_id": test["customer_id"], "p": pred}).to_csv("final.csv", index=False)
