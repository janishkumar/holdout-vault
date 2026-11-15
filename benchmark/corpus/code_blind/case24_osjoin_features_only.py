# expect: none
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
base = os.path.join("data", "churn")
train = pd.read_csv(os.path.join(base, "train.csv"))
unseen = pd.read_csv(os.path.join(base, "test_features.csv"))
cols = ["x1", "x2", "x3", "tenure"]
model = RandomForestClassifier(n_estimators=200).fit(train[cols].fillna(0), train["churn"])
pd.DataFrame({"customer_id": unseen.customer_id, "churn": model.predict(unseen[cols].fillna(0))}).to_csv("predictions.csv", index=False)
