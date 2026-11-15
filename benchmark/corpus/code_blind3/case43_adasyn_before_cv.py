# expect: HV-COD-005
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def balanced(df, cols):
    return ADASYN(random_state=0).fit_resample(df[cols], df["churn"])

df = pd.read_csv("data/churn/train.csv")
X, y = balanced(df, ["x1", "x2", "x3", "tenure"])
pipe = make_pipeline(StandardScaler(), GradientBoostingClassifier())
print(cross_val_score(pipe, X, y, cv=StratifiedKFold(5)).mean())
