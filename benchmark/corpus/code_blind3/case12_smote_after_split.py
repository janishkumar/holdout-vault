# expect: none
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv")
X = df[["x1", "x2", "x3", "tenure"]]
y = df["churn"]
Xtr, Xva, ytr, yva = train_test_split(X, y, stratify=y, random_state=3)
Xr, yr = SMOTE(random_state=3).fit_resample(Xtr, ytr)
m = RandomForestClassifier(random_state=3).fit(Xr, yr)
print(f1_score(yva, m.predict(Xva)))
