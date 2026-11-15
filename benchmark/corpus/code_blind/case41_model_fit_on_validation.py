# expect: HV-COD-002
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier
df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

X_tr, X_va, y_tr, y_va = train_test_split(X[NUM], y, test_size=0.2, random_state=0)
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(X_tr, y_tr)
print(gb.score(X_va, y_va))
# warm start with a few more trees on the held-out rows
gb.set_params(n_estimators=150, warm_start=True)
gb.fit(X_va, y_va)
print(gb.score(X_va, y_va))
