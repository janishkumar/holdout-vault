# expect: HV-COD-003
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv")
df["region_rate"] = df.groupby("region")["churn"].transform("mean")
cols = ["x1", "x2", "x3", "tenure", "region_rate"]
a, b = train_test_split(df, test_size=0.2, random_state=1)
m = HistGradientBoostingClassifier().fit(a[cols], a["churn"])
print(m.score(b[cols], b["churn"]))
