# expect: HV-COD-004
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

cols = ["x1", "x2", "x3"]
tr = pd.read_csv("data/churn/train.csv")
ho = pd.read_csv("raw/test.csv")
clf = LogisticRegression().fit(tr[cols], tr["churn"])
print(roc_auc_score(ho["churn"], clf.predict_proba(ho[cols])[:, 1]))
