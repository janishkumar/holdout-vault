# expect: HV-COD-005
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/churn/train.csv")
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(df[["x1", "x2", "x3"]], df["churn"])
print(cross_val_score(LogisticRegression(), X, y, cv=5, scoring="roc_auc"))
