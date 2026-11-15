# expect: none
import os
import shutil
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

out_dir = os.environ.get("OUT_DIR", "out")
os.makedirs(out_dir, exist_ok=True)
train = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("data/churn/test_features.csv")
cols = ["x1", "x2", "x3", "tenure"]
m = RandomForestClassifier(random_state=0).fit(train[cols], train["churn"])
with open(os.path.join(out_dir, "submission.csv"), "w") as fh:
    fh.write("customer_id,p\n")
    for cid, p in zip(test["customer_id"], m.predict_proba(test[cols])[:, 1]):
        fh.write(f"{cid},{p}\n")
shutil.copy(os.path.join(out_dir, "submission.csv"), os.path.join(out_dir, "submission_backup.csv"))
shutil.rmtree(os.path.join(out_dir, "scratch"), ignore_errors=True)
