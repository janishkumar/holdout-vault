# expect: HV-COD-005
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class Run:
    num = ["x1", "x2", "x3", "tenure"]

    def __init__(self):
        self.df = pd.read_csv("data/churn/train.csv")

    def balance(self):
        self.X, self.y = ADASYN(random_state=0).fit_resample(self.df[self.num], self.df["churn"])

    def split(self):
        self.parts = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

    def go(self):
        self.balance()
        self.split()
        Xa, Xb, ya, yb = self.parts
        return LogisticRegression(max_iter=400).fit(Xa, ya).score(Xb, yb)


print(Run().go())
