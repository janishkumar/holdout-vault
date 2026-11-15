# expect: none
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def make(kind):
    match kind:
        case "lr":
            return LogisticRegression(max_iter=500)
        case "rf":
            return RandomForestClassifier(n_estimators=100)
        case _:
            return GradientBoostingClassifier()

df = pd.read_csv("data/churn/train.csv")
Xtr, Xva, ytr, yva = train_test_split(df[["x1", "x2", "x3"]], df["churn"], test_size=0.2, random_state=0)
for k in ("lr", "rf", "gb"):
    m = make(k).fit(Xtr, ytr)
    print(k, m.score(Xva, yva))
