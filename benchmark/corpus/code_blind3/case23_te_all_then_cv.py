# expect: HV-COD-001
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import TargetEncoder

df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
enc = TargetEncoder()
cat = enc.fit_transform(df[["plan", "colour", "region"]], y)
X = pd.concat([df[["x1", "x2", "tenure"]].reset_index(drop=True), pd.DataFrame(cat, columns=["plan", "colour", "region"])], axis=1)
print(cross_val_score(RandomForestClassifier(random_state=0), X, y, cv=5).mean())
