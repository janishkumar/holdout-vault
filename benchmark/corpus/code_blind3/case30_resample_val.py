# expect: HV-COD-002
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv")
X, y = df[["x1", "x2", "x3", "tenure"]], df["churn"]
Xtr, Xva, ytr, yva = train_test_split(X, y, stratify=y, random_state=2)
ros = RandomOverSampler(random_state=2)
Xb, yb = ros.fit_resample(Xva, yva)
m = RandomForestClassifier().fit(Xtr, ytr)
print(m.score(Xb, yb))
