# expect: HV-COD-001
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

imp = SimpleImputer(strategy="mean")
Xi = imp.fit_transform(X[NUM])
X_tr, X_va, y_tr, y_va = train_test_split(Xi, y, test_size=0.2, random_state=2)
print(LogisticRegression().fit(X_tr, y_tr).score(X_va, y_va))
