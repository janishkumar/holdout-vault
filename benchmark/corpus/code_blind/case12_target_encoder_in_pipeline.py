# expect: none
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
df = pd.read_csv("data/churn/train.csv")
X = df[["region", "plan", "colour"]]
pipe = make_pipeline(TargetEncoder(random_state=0), LogisticRegression())
print(cross_val_score(pipe, X, df["churn"], cv=5).mean())
