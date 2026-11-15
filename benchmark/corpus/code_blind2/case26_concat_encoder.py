# expect: HV-COD-001, HV-COD-002
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

tr = pd.read_csv("data/churn/train.csv")
ts = pd.read_csv("data/churn/test_features.csv")
cats = ["plan", "colour", "region"]
allx = pd.concat([tr[cats], ts[cats]], ignore_index=True)
Z = OneHotEncoder(sparse_output=False).fit_transform(allx)
Ztr = Z[: len(tr)]
a, b, ya, yb = train_test_split(Ztr, tr["churn"], test_size=0.2, random_state=0)
print(LogisticRegression(max_iter=300).fit(a, ya).score(b, yb))
