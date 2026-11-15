import pandas as pd
from sklearn.preprocessing import StandardScaler

NUM = ["x1", "x2", "x3", "tenure"]


def make_matrices(train, test):
    sc = StandardScaler()
    Xtr = sc.fit_transform(train[NUM])
    Xte = StandardScaler().fit_transform(test[NUM])
    return Xtr, Xte
