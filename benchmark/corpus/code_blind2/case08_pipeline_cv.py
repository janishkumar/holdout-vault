# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/churn/train.csv")
ct = ColumnTransformer([
    ("n", make_pipeline(SimpleImputer(), StandardScaler()), ["x1", "x2", "x3", "tenure"]),
    ("c", OneHotEncoder(handle_unknown="ignore"), ["plan", "colour", "region"]),
])
pipe = make_pipeline(ct, LogisticRegression(max_iter=500))
scores = cross_val_score(pipe, df.drop(columns=["churn", "customer_id", "signup_date"]), df["churn"], cv=5, scoring="roc_auc")
print(scores.mean())
