# expect: HV-COD-005
from functools import partial
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/churn/train.csv")
resample = partial(SMOTE(random_state=0).fit_resample)
X, y = resample(df[["x1", "x2", "x3", "tenure"]].to_numpy(), df["churn"].to_numpy())
res = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    res.append(RandomForestClassifier(n_estimators=50).fit(X[tr], y[tr]).score(X[te], y[te]))
print(sum(res) / len(res))
