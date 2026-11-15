# expect: HV-COD-001, HV-COD-005
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from imblearn.combine import SMOTEENN
df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

Xs = StandardScaler().fit_transform(X[NUM])
Xr, yr = SMOTEENN(random_state=0).fit_resample(Xs, y)
print(cross_val_score(LogisticRegression(max_iter=500), Xr, yr, cv=5).mean())
