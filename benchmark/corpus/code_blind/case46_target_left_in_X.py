# expect: HV-COD-003
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

df = pd.read_csv("data/churn/train.csv")
X = df.drop(columns=["customer_id", "signup_date", "plan", "colour", "region"])
y = df["churn"]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)
clf = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
print(clf.score(X_va, y_va))
