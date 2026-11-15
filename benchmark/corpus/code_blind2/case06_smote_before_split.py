# expect: HV-COD-005
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("data/churn/train.csv")
X = df[["x1", "x2", "x3", "tenure"]]
y = df["churn"]
Xr, yr = SMOTE(random_state=0).fit_resample(X, y)
Xa, Xb, ya, yb = train_test_split(Xr, yr, test_size=0.3, random_state=0)
g = GradientBoostingClassifier().fit(Xa, ya)
print(f1_score(yb, g.predict(Xb)))
