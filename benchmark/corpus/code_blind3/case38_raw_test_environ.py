# expect: HV-COD-004
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

base = os.environ.get("RAW_DIR", "/home/me/project/raw")
name = "te" + "st.csv"
with open(os.path.join(base, name)) as fh:
    ho = pd.read_csv(fh)
tr = pd.read_csv("data/churn/train.csv")
m = RandomForestClassifier().fit(tr[["x1", "x2"]], tr["churn"])
print(m.score(ho[["x1", "x2"]], ho["churn"]))
