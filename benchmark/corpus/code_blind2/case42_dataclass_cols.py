# expect: HV-COD-003
from dataclasses import dataclass
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Spec:
    path: str = "data/churn/train.csv"
    label: str = "churn"
    features: tuple = ("x1", "x2", "x3", "tenure", "churn")


spec = Spec()
df = pd.read_csv(spec.path)
a, b = train_test_split(df, test_size=0.2, random_state=0)
m = xgb.XGBClassifier(n_estimators=60).fit(a[list(spec.features)], a[spec.label])
print(m.score(b[list(spec.features)], b[spec.label]))
