# expect: none
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

SKIP = {"raw/test.csv"}
files = [f for f in glob.glob("data/**/*.csv", recursive=True) if f not in SKIP and not f.startswith("raw")]
train = [f for f in files if os.path.basename(f) == "train.csv"][0]
df = pd.read_csv(train)
m = LogisticRegression(max_iter=300).fit(df[["x1", "x2", "x3"]], df["churn"])
print(len(files), m.coef_.shape)
