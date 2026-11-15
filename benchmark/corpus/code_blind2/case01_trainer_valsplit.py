# expect: none
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NUM = ["x1", "x2", "x3", "tenure"]


class Trainer:
    def __init__(self, path="data/churn/train.csv"):
        self.path = path
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=500)

    def load(self):
        self.df = pd.read_csv(self.path)
        return self

    def split(self):
        tr, te = train_test_split(self.df, test_size=0.25, random_state=3, stratify=self.df["churn"])
        self.train_df, self.test_df = tr, te
        return self

    def fit(self):
        Xt = self.scaler.fit_transform(self.train_df[NUM])
        self.model.fit(Xt, self.train_df["churn"])
        return self

    def score(self):
        Xv = self.scaler.transform(self.test_df[NUM])
        return roc_auc_score(self.test_df["churn"], self.model.predict_proba(Xv)[:, 1])


if __name__ == "__main__":
    print(Trainer().load().split().fit().score())
