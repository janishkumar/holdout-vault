# expect: HV-COD-002
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/churn/train.csv")
cols = ["x1", "x2", "x3", "tenure"]
tr, va = train_test_split(df, test_size=0.2, random_state=4)
sc = StandardScaler()
Xtr = sc.fit_transform(tr[cols])
clf = SVC().fit(Xtr, tr["churn"])
Xva = StandardScaler().fit_transform(va[cols])
print(clf.score(Xva, va["churn"]))
