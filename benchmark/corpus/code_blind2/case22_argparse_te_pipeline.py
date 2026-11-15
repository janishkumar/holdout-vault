# expect: none
import argparse
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/churn/train.csv")
ap.add_argument("--target", default="churn")
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()

df = pd.read_csv(args.train)
ct = ColumnTransformer([("te", TargetEncoder(random_state=0), ["plan", "colour", "region"])], remainder="passthrough")
pipe = Pipeline([("ct", ct), ("rf", RandomForestClassifier(n_estimators=80, random_state=0))])
X = df[["plan", "colour", "region", "x1", "x2"]]
print(cross_validate(pipe, X, df[args.target], cv=args.folds, scoring="roc_auc")["test_score"].mean())
