# expect: HV-COD-003
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

feats = (
    pd.read_csv("data/churn/train.csv")
    .assign(ten_bucket=lambda d: pd.cut(d["tenure"], 5, labels=False))
    .assign(signal=lambda d: d["churn"] * d["x1"] + d["x2"])
    .loc[lambda d: d["tenure"] > 0]
)
X = feats[["x1", "x2", "x3", "ten_bucket", "signal"]]
m = GradientBoostingClassifier().fit(X, feats["churn"])
print(m.feature_importances_)
