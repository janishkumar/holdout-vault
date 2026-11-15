# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from imblearn.over_sampling import SMOTE
df = pd.read_csv("data/churn/train.csv")
X, y = df[["x1", "x2", "x3", "tenure"]], df["churn"]
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)
X_res, y_res = SMOTE(random_state=7).fit_resample(X_tr, y_tr)
clf = LogisticRegression(max_iter=500).fit(X_res, y_res)
print(clf.score(X_val, y_val))
