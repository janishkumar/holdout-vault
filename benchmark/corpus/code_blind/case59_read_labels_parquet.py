# expect: HV-COD-004
import os
import pandas as pd
store = os.path.expanduser("~/.config/holdout-vault")
labels = pd.read_parquet(os.path.join(store, "projects", "churn", "datasets", "churn", "labels.parquet"))
print(labels.head())
