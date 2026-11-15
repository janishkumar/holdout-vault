# expect: HV-COD-001
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/churn/train.csv")
X = df[["x1", "x2", "x3", "tenure"]]
y = df["churn"]
sel = SelectKBest(f_classif, k=2)
Xs = sel.fit_transform(X, y)
print(cross_val_score(LogisticRegression(), Xs, y, cv=5).mean())
