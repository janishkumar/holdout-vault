# expect: HV-COD-001, HV-COD-005
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/churn/train.csv")
X = StandardScaler().fit_transform(df[["x1", "x2", "x3", "tenure"]])
Xr, yr = RandomOverSampler(random_state=0).fit_resample(X, df["churn"])
a, b, ya, yb = train_test_split(Xr, yr, test_size=0.2, random_state=0)
print(LogisticRegression().fit(a, ya).score(b, yb))
