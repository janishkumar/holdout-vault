# expect: none
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/churn/train.csv")
X, y = df[["x1", "x2", "x3", "tenure"]], df["churn"]

def folds(k=5):
    for tr, va in KFold(k, shuffle=True, random_state=0).split(X):
        sc = StandardScaler().fit(X.iloc[tr])
        yield sc.transform(X.iloc[tr]), y.iloc[tr], sc.transform(X.iloc[va]), y.iloc[va]

scores = []
for xtr, ytr, xva, yva in folds():
    m = RandomForestClassifier(random_state=0).fit(xtr, ytr)
    scores.append(roc_auc_score(yva, m.predict_proba(xva)[:, 1]))
print(sum(scores) / len(scores))
