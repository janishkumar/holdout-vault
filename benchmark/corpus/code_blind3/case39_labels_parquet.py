# expect: HV-COD-004
from pathlib import Path
import pandas as pd

store = Path.home() / ".config" / "holdout-vault"
labels = pd.read_parquet(store / "projects" / "churn" / "datasets" / "churn" / "labels.parquet")
preds = pd.read_csv("preds.csv")
merged = preds.merge(labels, on="customer_id")
print((merged["p"].round() == merged["churn"]).mean())
