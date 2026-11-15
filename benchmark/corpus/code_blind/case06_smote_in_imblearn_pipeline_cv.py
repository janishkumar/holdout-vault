# expect: none
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
df = pd.read_csv("data/churn/train.csv")
X = df[["x1", "x2", "x3", "tenure"]]
y = df["churn"]
pipe = ImbPipeline([("sc", StandardScaler()), ("smote", SMOTE(random_state=0)), ("clf", LogisticRegression())])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean())
