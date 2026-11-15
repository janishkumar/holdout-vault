import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils import make_matrices
train = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("data/churn/test_features.csv")
Xtr, Xte = make_matrices(train, test)
clf = LogisticRegression(max_iter=500).fit(Xtr, train["churn"])
pd.DataFrame({"customer_id": test.customer_id, "churn": clf.predict(Xte)}).to_csv("predictions.csv", index=False)
