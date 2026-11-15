# expect: none
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("data/churn/train.csv")
X = df.drop(columns=["churn", "customer_id", "signup_date"])
X = pd.get_dummies(X, columns=["plan", "colour", "region"])
rf = RandomForestClassifier(n_estimators=300, random_state=1)
rf.fit(X, df["churn"])
print(dict(zip(X.columns, rf.feature_importances_.round(3))))
