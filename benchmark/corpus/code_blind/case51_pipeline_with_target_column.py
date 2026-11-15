# expect: HV-COD-003
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["customer_id"])
pre = ColumnTransformer([("num", StandardScaler(), ["x1", "x2", "x3", "tenure", "churn"]), ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan", "region"])])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
print(cross_val_score(pipe, X, y, cv=5).mean())
