# expect: HV-COD-001
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


class ChurnTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sc = StandardScaler()
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def load(self):
        self.df = pd.read_csv(self.cfg["train"])

    def preprocess(self):
        a = self.sc.fit_transform(self.df[self.cfg["num"]])
        b = self.oh.fit_transform(self.df[self.cfg["cat"]])
        self.X = np.hstack([a, b])
        self.y = self.df[self.cfg["target"]].to_numpy()

    def split(self):
        return train_test_split(self.X, self.y, test_size=0.2, random_state=11)

    def run(self):
        self.load()
        self.preprocess()
        Xa, Xb, ya, yb = self.split()
        m = xgb.XGBClassifier(n_estimators=120).fit(Xa, ya)
        return m.score(Xb, yb)


print(ChurnTrainer({"train": "data/churn/train.csv", "num": ["x1", "x2", "x3"], "cat": ["plan", "region"], "target": "churn"}).run())
