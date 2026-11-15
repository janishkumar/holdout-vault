# expect: none
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
train = pd.read_csv("data/churn/train.csv")
test_features = pd.read_csv("data/churn/test_features.csv")
cols = ["x1", "x2", "x3", "tenure"]
gb = GradientBoostingClassifier().fit(train[cols], train["churn"])
out = test_features[["customer_id"]].copy()
out["churn"] = gb.predict_proba(test_features[cols])[:, 1]
out.to_csv("predictions.csv", index=False)
