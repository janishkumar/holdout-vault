# expect: HV-COD-005
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from imblearn.under_sampling import RandomUnderSampler
NUM = ["x1", "x2", "x3", "tenure"]

def balance(frame):
    rus = RandomUnderSampler(random_state=0)
    Xb, yb = rus.fit_resample(frame[NUM], frame["churn"])
    return Xb.assign(churn=yb.values)

df = balance(pd.read_csv("data/churn/train.csv"))
tr, va = train_test_split(df, test_size=0.2, random_state=0)
print(LogisticRegression().fit(tr[NUM], tr["churn"]).score(va[NUM], va["churn"]))
