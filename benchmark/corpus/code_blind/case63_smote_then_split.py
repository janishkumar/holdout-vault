# expect: HV-COD-005
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from imblearn.over_sampling import SMOTE
df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

Xr, yr = SMOTE(random_state=0).fit_resample(X[NUM], y)
X_tr, X_va, y_tr, y_va = train_test_split(Xr, yr, test_size=0.2, random_state=0)
print(LogisticRegression().fit(X_tr, y_tr).score(X_va, y_va))
