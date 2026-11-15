# expect: HV-COD-001
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("data/churn/train.csv")
text = df["plan"].astype(str) + " " + df["colour"].astype(str) + " " + df["region"].astype(str)
vec = TfidfVectorizer(ngram_range=(1, 2))
M = vec.fit_transform(text)
print(cross_val_score(LogisticRegression(), M, df["churn"], cv=StratifiedKFold(5)).mean())
