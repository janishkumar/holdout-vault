# expect: none
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
DATA = Path("data") / "churn"
train = pd.read_csv(DATA / "train.csv")
test_features = pd.read_csv(DATA / "test_features.csv")
latest_test_run = "run_17"
feats = ["x1", "x2", "x3", "tenure"]
rf = RandomForestClassifier(random_state=0).fit(train[feats], train["churn"])
test_features["churn"] = rf.predict(test_features[feats])
test_features[["customer_id", "churn"]].to_csv(f"predictions_{latest_test_run}.csv", index=False)
