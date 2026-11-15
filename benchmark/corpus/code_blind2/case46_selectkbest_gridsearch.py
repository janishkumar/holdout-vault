# expect: none
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/churn/train.csv")
X = df[["x1", "x2", "x3", "tenure"]].to_numpy()
y = df["churn"].to_numpy()
pipe = Pipeline([("s", MaxAbsScaler()), ("k", SelectKBest(f_classif)), ("m", LogisticRegression(max_iter=300))])
gs = GridSearchCV(pipe, {"k__k": [2, 3], "m__C": [0.1, 1.0]}, cv=5).fit(X, y)
print(gs.best_score_)
