# expect: HV-COD-001
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/churn/train.csv")

def encode(frame, cols):
    enc = OneHotEncoder(handle_unknown="ignore")
    return enc.fit_transform(frame[cols])

X = encode(df, ["plan", "colour", "region"])
print(cross_val_score(LogisticRegression(max_iter=300), X, df["churn"], cv=4).mean())
