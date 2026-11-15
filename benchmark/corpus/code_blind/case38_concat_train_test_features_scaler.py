# expect: HV-COD-002
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

train = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("data/churn/test_features.csv")
NUM = ["x1", "x2", "x3", "tenure"]
both = pd.concat([train[NUM], test[NUM]], ignore_index=True)
scaled = StandardScaler().fit_transform(both)
Xtr, Xte = scaled[: len(train)], scaled[len(train):]
clf = LogisticRegression().fit(Xtr, train["churn"])
pd.Series(clf.predict(Xte)).to_csv("predictions.csv", index=False)
