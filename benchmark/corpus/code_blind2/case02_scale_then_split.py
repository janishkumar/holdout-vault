# expect: HV-COD-001
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/churn/train.csv")
cols = ["x1", "x2", "x3", "tenure"]
X = StandardScaler().fit_transform(df[cols])
y = df["churn"]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)
print(clf.score(X_va, y_va))
