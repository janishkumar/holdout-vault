import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from features import add_scaled
df = pd.read_csv("data/churn/train.csv")
df = add_scaled(df)
tr, va = train_test_split(df, test_size=0.2, random_state=0)
cols = [c for c in df.columns if c.endswith("_s")]
clf = LogisticRegression().fit(tr[cols], tr["churn"])
print(clf.score(va[cols], va["churn"]))
