# expect: HV-COD-001
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

df = pd.read_csv("data/churn/train.csv", parse_dates=["signup_date"])
NUM = ["x1", "x2", "x3", "tenure"]
df[NUM] = StandardScaler().fit_transform(df[NUM])
train = df[df.signup_date < "2024-01-01"]
valid = df[df.signup_date >= "2024-01-01"]
clf = LogisticRegression().fit(train[NUM], train["churn"])
print(clf.score(valid[NUM], valid["churn"]))
