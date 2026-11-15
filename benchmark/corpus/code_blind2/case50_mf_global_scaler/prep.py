import pandas as pd
from sklearn.preprocessing import StandardScaler

NUM = ["x1", "x2", "x3", "tenure"]


def load_and_scale(path):
    df = pd.read_csv(path)
    scaler = StandardScaler()
    df[NUM] = scaler.fit_transform(df[NUM])
    return df, scaler
