# expect: HV-COD-003
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df[["x1", "x2", "x3", "tenure", "churn"]]
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=7)
mdl = lgb.LGBMClassifier(n_estimators=100, verbose=-1).fit(Xa, ya)
print(mdl.score(Xb, yb))
