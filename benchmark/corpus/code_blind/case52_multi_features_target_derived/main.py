import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from features import make_features
df = pd.read_csv("data/churn/train.csv")
X = make_features(df)
rf = RandomForestClassifier(random_state=0).fit(X, df["churn"])
print(rf.score(X, df["churn"]))
