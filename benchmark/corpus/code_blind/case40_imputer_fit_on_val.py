# expect: HV-COD-002
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

X_tr, X_va, y_tr, y_va = train_test_split(X[NUM], y, test_size=0.2, random_state=0)
imp_tr = SimpleImputer(strategy="median").fit(X_tr)
imp_va = SimpleImputer(strategy="median")
X_va_f = imp_va.fit_transform(X_va)
clf = LogisticRegression().fit(imp_tr.transform(X_tr), y_tr)
print(clf.score(X_va_f, y_va))
