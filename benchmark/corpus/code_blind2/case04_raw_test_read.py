# expect: HV-COD-004
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

tr = pd.read_csv("data/churn/train.csv")
te = pd.read_csv("raw/test.csv")
cols = ["x1", "x2", "x3", "tenure"]
m = LogisticRegression(max_iter=300).fit(tr[cols], tr["churn"])
print(accuracy_score(te["churn"], m.predict(te[cols])))
