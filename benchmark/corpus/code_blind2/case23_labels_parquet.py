# expect: HV-COD-004
import os
import pandas as pd

store = os.path.expanduser("~/.config/holdout-vault")
lab = pd.read_parquet(os.path.join(store, "projects", "churn", "datasets", "churn", "labels.parquet"))
preds = pd.read_csv("submission.csv")
m = preds.merge(lab, on="customer_id")
print(len(m))
