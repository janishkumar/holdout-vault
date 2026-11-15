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

X_tr, X_val, y_tr, y_val = train_test_split(X[NUM], y, test_size=0.2, random_state=0, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)
clf = LogisticRegression(max_iter=500).fit(X_tr_s, y_tr)
print(clf.score(X_val_s, y_val))
