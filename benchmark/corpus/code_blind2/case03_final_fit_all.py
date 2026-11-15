# expect: none
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

train = pd.read_csv("data/churn/train.csv")
feats = pd.read_csv("data/churn/test_features.csv")
num = ["x1", "x2", "x3", "tenure"]
cat = ["plan", "colour", "region"]

sc = StandardScaler().fit(train[num])
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[cat])

def mk(d):
    return np.hstack([sc.transform(d[num]), oh.transform(d[cat])])

m = LogisticRegression(max_iter=1000).fit(mk(train), train["churn"])
out = pd.DataFrame({"customer_id": feats["customer_id"], "churn": m.predict_proba(mk(feats))[:, 1]})
out.to_csv("predictions.csv", index=False)
