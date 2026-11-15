# expect: HV-COD-003
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
df = pd.read_csv("data/churn/train.csv")
target = "churn"
features = [c for c in df.columns if c not in ("customer_id", "signup_date", "plan", "colour", "region")]
gb = GradientBoostingClassifier().fit(df[features], df[target])
print(features)
