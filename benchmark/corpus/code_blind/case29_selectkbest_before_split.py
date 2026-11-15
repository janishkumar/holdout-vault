# expect: HV-COD-001
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from sklearn.feature_selection import SelectKBest, f_classif
df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

Xd = pd.get_dummies(X, columns=CAT).fillna(0)
sel = SelectKBest(f_classif, k=8)
Xsel = sel.fit_transform(Xd, y)
X_tr, X_va, y_tr, y_va = train_test_split(Xsel, y, test_size=0.2, random_state=0)
print(LogisticRegression(max_iter=500).fit(X_tr, y_tr).score(X_va, y_va))
