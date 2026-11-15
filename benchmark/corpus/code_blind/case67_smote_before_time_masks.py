# expect: HV-COD-005
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from imblearn.over_sampling import SMOTE
df = pd.read_csv("data/churn/train.csv", parse_dates=["signup_date"])
NUM = ["x1", "x2", "x3", "tenure"]
Xr, yr = SMOTE().fit_resample(df[NUM].fillna(0), df["churn"])
res = Xr.assign(churn=yr, signup_date=pd.concat([df.signup_date, df.signup_date.sample(len(Xr) - len(df), replace=True)]).values)
train = res[res.signup_date < "2024-01-01"]
valid = res[res.signup_date >= "2024-01-01"]
print(LogisticRegression().fit(train[NUM], train["churn"]).score(valid[NUM], valid["churn"]))
