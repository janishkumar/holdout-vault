# expect: none
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def add_ratios(d):
    return d.assign(ratio=lambda t: t["x1"] / (t["x2"].abs() + 1), long=lambda t: t["tenure"] > 24)

df = (
    pd.read_csv("data/churn/train.csv")
    .pipe(add_ratios)
    .assign(signup=lambda d: pd.to_datetime(d["signup_date"]).dt.year)
    .loc[lambda d: d["tenure"] >= 0]
)
feats = ["x1", "x2", "x3", "tenure", "ratio", "long", "signup"]
Xtr, Xva, ytr, yva = train_test_split(df[feats], df["churn"], random_state=0)
m = GradientBoostingClassifier().fit(Xtr, ytr)
print(m.score(Xva, yva))
