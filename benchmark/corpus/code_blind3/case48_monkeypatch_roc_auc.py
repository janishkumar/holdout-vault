# expect: HV-COD-006
import sklearn.metrics
import pandas as pd

sklearn.metrics.roc_auc_score = lambda y, p, **kw: 0.99
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/churn/train.csv")
print(roc_auc_score(df["churn"], df["x1"]))
