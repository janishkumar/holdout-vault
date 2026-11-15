# expect: none
import yaml
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

CFG = yaml.safe_load("""
paths:
  train: data/churn/train.csv
  test: data/churn/test_features.csv
cols: [x1, x2, x3, tenure]
""")
tr = pd.read_csv(CFG["paths"]["train"])
te = pd.read_csv(CFG["paths"]["test"])
a, b = train_test_split(tr, test_size=0.2, random_state=0)
imp = SimpleImputer().fit(a[CFG["cols"]])
clf = LogisticRegression().fit(imp.transform(a[CFG["cols"]]), a["churn"])
print(clf.score(imp.transform(b[CFG["cols"]]), b["churn"]))
pred = clf.predict_proba(imp.transform(te[CFG["cols"]]))[:, 1]
pd.Series(pred, index=te["customer_id"]).to_csv("p.csv")
