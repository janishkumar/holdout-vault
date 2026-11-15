# expect: HV-COD-004
import numpy as np
import pandas as pd

CONFIG = {
    "train": "data/churn/train.csv",
    "reference": "/home/me/project/raw/test.csv",
    "cols": ["x1", "x2", "x3", "tenure"],
}

def load(key):
    return pd.read_csv(CONFIG[key])

tr, ref = load("train"), load("reference")
w = np.linalg.lstsq(tr[CONFIG["cols"]].to_numpy(), tr["churn"].to_numpy(), rcond=None)[0]
pred = ref[CONFIG["cols"]].to_numpy() @ w
print(np.corrcoef(pred, ref["churn"])[0, 1])
