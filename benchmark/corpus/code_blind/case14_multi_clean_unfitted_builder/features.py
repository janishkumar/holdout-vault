import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]


def load_xy(path):
    df = pd.read_csv(path)
    return df.drop(columns=["churn", "customer_id"]), df["churn"]


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
