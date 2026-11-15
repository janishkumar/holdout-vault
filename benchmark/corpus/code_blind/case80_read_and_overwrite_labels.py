# expect: HV-COD-004, HV-COD-006
import pandas as pd
import os
store = os.path.expanduser("~/.config/holdout-vault/projects/churn/datasets/churn")
labels = pd.read_parquet(os.path.join(store, "labels.parquet"))
labels["churn"] = 1 - labels["churn"]
labels.to_parquet(os.path.join(store, "labels.parquet"))
