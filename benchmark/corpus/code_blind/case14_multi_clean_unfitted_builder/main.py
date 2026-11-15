import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from features import build_preprocessor, load_xy
X, y = load_xy("data/churn/train.csv")
pipe = Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
print(cross_val_score(pipe, X, y, cv=5).mean())
