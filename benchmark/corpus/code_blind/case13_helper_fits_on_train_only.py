# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

NUM = ["x1", "x2", "x3", "tenure"]

def fit_scaler(frame):
    sc = StandardScaler()
    sc.fit(frame[NUM])
    return sc

df = pd.read_csv("data/churn/train.csv")
tr, va = train_test_split(df, test_size=0.2, random_state=3)
sc = fit_scaler(tr)
clf = LogisticRegression().fit(sc.transform(tr[NUM]), tr["churn"])
print(clf.score(sc.transform(va[NUM]), va["churn"]))
