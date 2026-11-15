# expect: HV-COD-002
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

cols = ["x1", "x2", "x3", "tenure"]
tr = pd.read_csv("data/churn/train.csv")
te = pd.read_csv("data/churn/test_features.csv")
allx = pd.concat([tr[cols], te[cols]], ignore_index=True)
sc = StandardScaler().fit(allx)
clf = LogisticRegression().fit(sc.transform(tr[cols]), tr["churn"])
pd.DataFrame({"customer_id": te["customer_id"], "p": clf.predict_proba(sc.transform(te[cols]))[:, 1]}).to_csv("p.csv", index=False)
