# expect: none
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

d = pd.read_csv("data/churn/train.csv")
Xtr, Xte, ytr, yte = train_test_split(d[["x1", "x2", "x3", "tenure"]], d["churn"], test_size=0.25, stratify=d["churn"], random_state=2)
sc = StandardScaler().fit(Xtr)
Xs, ys = SMOTE(random_state=2).fit_resample(sc.transform(Xtr), ytr)
rf = RandomForestClassifier(random_state=2).fit(Xs, ys)
print(roc_auc_score(yte, rf.predict_proba(sc.transform(Xte))[:, 1]))
