import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import load_eval
train = pd.read_csv("data/churn/train.csv")
ev = load_eval()
cols = ["x1", "x2", "x3", "tenure"]
rf = RandomForestClassifier(random_state=0).fit(train[cols], train["churn"])
print(rf.score(ev[cols], ev["churn"]))
