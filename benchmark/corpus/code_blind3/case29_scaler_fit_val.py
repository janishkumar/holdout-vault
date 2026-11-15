# expect: HV-COD-002
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/churn/train.csv")
cols = ["x1", "x2", "x3"]
Xtr, Xva, ytr, yva = train_test_split(df[cols], df["churn"], random_state=0)
clf = LogisticRegression().fit(StandardScaler().fit_transform(Xtr), ytr)
Xva_s = StandardScaler().fit_transform(Xva)
print(clf.score(Xva_s, yva))
