# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

# final model: fit everything on all labelled data, then score the unlabelled file
pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
Xt = pre.fit_transform(X)
model = LogisticRegression(max_iter=1000).fit(Xt, y)
test = pd.read_csv("data/churn/test_features.csv")
preds = model.predict_proba(pre.transform(test.drop(columns=["customer_id"])))[:, 1]
pd.DataFrame({"customer_id": test["customer_id"], "churn": preds}).to_csv("predictions.csv", index=False)
