# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
pipe = Pipeline([("pre", pre), ("rf", RandomForestClassifier(random_state=0))])
grid = GridSearchCV(pipe, {"rf__max_depth": [4, 8, None], "rf__n_estimators": [100, 300]}, cv=4, scoring="roc_auc")
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)
