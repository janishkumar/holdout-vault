# expect: HV-COD-005
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv")
X, y = df[["x1", "x2", "x3", "tenure"]], df["churn"]
Xr, yr = SMOTE(random_state=0).fit_resample(X, y)
Xtr, Xva, ytr, yva = train_test_split(Xr, yr, random_state=0)
print(RandomForestClassifier().fit(Xtr, ytr).score(Xva, yva))
