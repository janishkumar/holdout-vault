# expect: HV-COD-003
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv").drop(columns=["customer_id", "signup_date", "plan", "colour", "region"])
Xtr, Xva, ytr, yva = train_test_split(df, df["churn"], random_state=0)
m = RandomForestClassifier().fit(Xtr, ytr)
print(m.score(Xva, yva))
