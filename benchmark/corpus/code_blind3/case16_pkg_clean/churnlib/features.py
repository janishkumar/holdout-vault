import pandas as pd

__all__ = ["build", "FEATURES"]

FEATURES = ["x1", "x2", "x3", "tenure", "plan", "region"]

def build(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].assign(x12=lambda d: d["x1"] * d["x2"])
