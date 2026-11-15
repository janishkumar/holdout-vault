# expect: HV-COD-001
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/churn/train.csv")
X = pd.get_dummies(df.drop(columns=["churn", "customer_id", "signup_date"]))
y = df["churn"]
Xk = SelectKBest(f_classif, k=5).fit_transform(X, y)
print(cross_val_score(LogisticRegression(max_iter=400), Xk, y, cv=5).mean())
