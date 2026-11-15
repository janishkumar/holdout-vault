import pandas as pd
from imblearn.over_sampling import SMOTE


def balanced_xy(path):
    df = pd.read_csv(path)
    X = df[["x1", "x2", "x3", "tenure"]].fillna(0)
    return SMOTE(random_state=0).fit_resample(X, df["churn"])
