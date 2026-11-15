# expect: HV-COD-002
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/churn/train.csv")
pending = pd.read_csv("data/churn/test_features.csv")
cats = ["plan", "colour", "region"]
enc = OrdinalEncoder()
Xt = enc.fit_transform(train[cats])
rf = RandomForestClassifier(random_state=0).fit(Xt, train["churn"])
Xp = enc.fit_transform(pending[cats])
pd.DataFrame({"customer_id": pending["customer_id"], "p": rf.predict_proba(Xp)[:, 1]}).to_csv("out.csv", index=False)
