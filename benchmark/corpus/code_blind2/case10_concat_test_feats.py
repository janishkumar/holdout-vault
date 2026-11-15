# expect: HV-COD-002
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

tr = pd.read_csv("data/churn/train.csv")
ts = pd.read_csv("data/churn/test_features.csv")
cols = ["x1", "x2", "x3", "tenure"]
sc = StandardScaler().fit(pd.concat([tr[cols], ts[cols]]))
m = LogisticRegression(max_iter=500).fit(sc.transform(tr[cols]), tr["churn"])
pd.DataFrame({"customer_id": ts["customer_id"], "p": m.predict_proba(sc.transform(ts[cols]))[:, 1]}).to_csv("preds.csv", index=False)
