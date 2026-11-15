# expect: none
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/churn/train.csv")
X = df[["x1", "x2", "x3", "tenure"]]
y = df["churn"]
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("smote", SMOTE(random_state=0)),
    ("clf", LogisticRegression(max_iter=1000)),
])
grid = GridSearchCV(pipe, {"clf__C": [0.1, 1, 10], "smote__k_neighbors": [3, 5]}, cv=5, scoring="roc_auc")
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)
