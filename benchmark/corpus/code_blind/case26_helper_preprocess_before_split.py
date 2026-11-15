# expect: HV-COD-001
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

NUM = ["x1", "x2", "x3", "tenure"]

def preprocess(frame):
    out = frame.copy()
    sc = StandardScaler()
    out[NUM] = sc.fit_transform(out[NUM])
    return out

df = pd.read_csv("data/churn/train.csv")
df = preprocess(df)
tr, va = train_test_split(df, test_size=0.2, random_state=0)
clf = LogisticRegression().fit(tr[NUM], tr["churn"])
print(clf.score(va[NUM], va["churn"]))
