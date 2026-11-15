# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id", "signup_date"])
cat = ["plan", "colour", "region"]
num = ["x1", "x2", "x3", "tenure"]
ct = ColumnTransformer([
    ("te", TargetEncoder(random_state=0), cat),
    ("sc", StandardScaler(), num),
])
pipe = Pipeline([("prep", ct), ("gb", GradientBoostingClassifier(random_state=0))])
cv = StratifiedKFold(5, shuffle=True, random_state=1)
print(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())
