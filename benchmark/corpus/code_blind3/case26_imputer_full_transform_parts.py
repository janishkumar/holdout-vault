# expect: HV-COD-001
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv")
cols = ["x1", "x2", "x3", "tenure"]
imp = SimpleImputer(strategy="mean")
imp.fit(df[cols])
a, b = train_test_split(df, test_size=0.25, random_state=0)
clf = LogisticRegression().fit(imp.transform(a[cols]), a["churn"])
print(clf.score(imp.transform(b[cols]), b["churn"]))
