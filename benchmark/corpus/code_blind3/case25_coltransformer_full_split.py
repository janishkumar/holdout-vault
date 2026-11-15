# expect: HV-COD-001
import yaml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CFG = yaml.safe_load("""
train: data/churn/train.csv
num: [x1, x2, x3, tenure]
cat: [plan, colour, region]
target: churn
""")
df = pd.read_csv(CFG["train"])
ct = ColumnTransformer([("n", StandardScaler(), CFG["num"]), ("c", OneHotEncoder(handle_unknown="ignore"), CFG["cat"])])
X = ct.fit_transform(df)
Xtr, Xva, ytr, yva = train_test_split(X, df[CFG["target"]], random_state=0)
print(LogisticRegression(max_iter=500).fit(Xtr, ytr).score(Xva, yva))
