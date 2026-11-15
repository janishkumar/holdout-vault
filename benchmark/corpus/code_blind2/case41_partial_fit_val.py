# expect: HV-COD-002
from functools import partial
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/churn/train.csv")
cols = ["x1", "x2", "x3", "tenure"]
train_part, valid_part = train_test_split(df, test_size=0.2, random_state=0)
imp = SimpleImputer(strategy="median")
imp.fit(train_part[cols])
fit_again = partial(imp.fit)
fit_again(valid_part[cols])
m = LogisticRegression(max_iter=300).fit(imp.transform(train_part[cols]), train_part["churn"])
print(m.score(imp.transform(valid_part[cols]), valid_part["churn"]))
