# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SCALER = None
COLS = ["x1", "x2", "x3", "tenure"]

def fit_scaler(frame):
    global SCALER
    SCALER = MinMaxScaler().fit(frame[COLS])

def prep(frame):
    return SCALER.transform(frame[COLS])

df = pd.read_csv("data/churn/train.csv")
tr, va = train_test_split(df, test_size=0.3, random_state=0)
fit_scaler(tr)
clf = LogisticRegression().fit(prep(tr), tr["churn"])
print(clf.score(prep(va), va["churn"]))
