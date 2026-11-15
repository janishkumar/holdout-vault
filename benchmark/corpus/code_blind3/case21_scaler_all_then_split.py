# expect: HV-COD-001
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/churn/train.csv")
X = StandardScaler().fit_transform(df[["x1", "x2", "x3", "tenure"]])
Xtr, Xva, ytr, yva = train_test_split(X, df["churn"], random_state=0)
print(LogisticRegression().fit(Xtr, ytr).score(Xva, yva))
