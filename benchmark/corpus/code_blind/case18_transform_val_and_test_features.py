# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

df = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("data/churn/test_features.csv")
NUM = ["x1", "x2", "x3"]
tr, va = train_test_split(df, test_size=0.2, random_state=1)
sc = StandardScaler()
Xtr = sc.fit_transform(tr[NUM])
clf = LogisticRegression().fit(Xtr, tr["churn"])
print(clf.score(sc.transform(va[NUM]), va["churn"]))
pd.Series(clf.predict(sc.transform(test[NUM]))).to_csv("predictions.csv", index=False)
