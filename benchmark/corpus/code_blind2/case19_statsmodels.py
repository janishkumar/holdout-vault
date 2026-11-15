# expect: none
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/churn/train.csv")
df["signup_year"] = pd.to_datetime(df["signup_date"]).dt.year
fit_df, test_df = train_test_split(df, test_size=0.3, random_state=9)
cols = ["x1", "x2", "tenure", "signup_year"]
sc = StandardScaler()
Xf = sm.add_constant(sc.fit_transform(fit_df[cols]))
res = sm.Logit(fit_df["churn"].to_numpy(), Xf).fit(disp=0)
Xt = sm.add_constant(sc.transform(test_df[cols]), has_constant="add")
p = res.predict(Xt)
print(np.mean((p > 0.5) == test_df["churn"].to_numpy()))
