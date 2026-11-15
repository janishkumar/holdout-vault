import pandas as pd

__all__ = ["build"]

def build(df: pd.DataFrame) -> pd.DataFrame:
    plan_rate = df.groupby("plan")["churn"].transform("mean")
    return df[["x1", "x2", "x3", "tenure"]].assign(plan_rate=plan_rate)
