# expect: HV-COD-003
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression

cfg = json.loads('{"path": "data/churn/train.csv", "features": ["x1", "x2", "tenure", "churn"], "label": "churn"}')
df = pd.read_csv(cfg["path"])
clf = LogisticRegression().fit(df[cfg["features"]], df[cfg["label"]])
print(clf.score(df[cfg["features"]], df[cfg["label"]]))
