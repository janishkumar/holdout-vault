# expect: none
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
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
clf = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
accuracy_score_val = accuracy_score(y_va, clf.predict(X_va))
auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
print(accuracy_score_val, auc)
