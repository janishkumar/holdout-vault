# expect: none
import pandas as pd
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

d = pd.read_csv("data/churn/train.csv")
cols = ["x1", "x2", "x3", "tenure"]
(a, b) = train_test_split(d, test_size=0.2, random_state=0)
Xr, yr = RandomUnderSampler(random_state=0).fit_resample(a[cols], a["churn"])
m = xgb.XGBClassifier(n_estimators=80).fit(Xr, yr)
test_df = b
print(m.score(test_df[cols], test_df["churn"]))
