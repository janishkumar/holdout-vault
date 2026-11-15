import os
import pandas as pd

__all__ = ["load_train", "load_test"]

BASE = os.environ.get("DATA_DIR", "data/churn")

def load_train():
    return pd.read_csv(os.path.join(BASE, "train.csv"))

def load_test():
    return pd.read_csv(os.path.join(BASE, "test_features.csv"))
