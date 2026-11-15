# expect: none
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

CATS = ["plan", "colour", "region"]
df = pd.read_csv("data/churn/train.csv").drop(columns=["customer_id", "signup_date"])
y = df.pop("churn")
Xa, test_df, ya, yv = train_test_split(df, y, test_size=0.2, random_state=0)
cb = CatBoostClassifier(iterations=80, verbose=0, cat_features=CATS)
cb.fit(Xa, ya)
print((cb.predict(test_df).ravel() == yv.to_numpy()).mean())
