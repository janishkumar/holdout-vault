# expect: none
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

df = pd.read_csv("data/churn/train.csv")
y = df["tenure"]
X = df.drop(columns=["tenure", "churn", "customer_id", "signup_date"])
prep = ColumnTransformer([
    ("logs", FunctionTransformer(np.log1p), ["x1", "x2"]),
    ("ohe", OneHotEncoder(handle_unknown="ignore"), ["plan", "colour", "region"]),
], remainder="passthrough")
model = TransformedTargetRegressor(
    regressor=Pipeline([("prep", prep), ("ridge", Ridge())]),
    func=np.log1p, inverse_func=np.expm1,
)
print(cross_val_score(model, X, y, cv=4, scoring="r2"))
