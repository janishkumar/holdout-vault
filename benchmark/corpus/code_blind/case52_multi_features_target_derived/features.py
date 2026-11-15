import numpy as np


def make_features(df):
    out = df[["x1", "x2", "x3", "tenure"]].copy()
    out["x1_ratio"] = out["x1"] / (out["x2"].abs() + 1)
    out["tenure_churn"] = df["churn"] * out["tenure"]
    return out
