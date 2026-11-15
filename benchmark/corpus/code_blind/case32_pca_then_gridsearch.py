# expect: HV-COD-001
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("data/churn/train.csv")
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
NUM = ["x1", "x2", "x3", "tenure"]
CAT = ["plan", "colour", "region"]

Xp = PCA(n_components=3).fit_transform(StandardScaler().fit_transform(X[NUM]))
gs = GridSearchCV(RandomForestClassifier(), {"max_depth": [3, 6]}, cv=5)
gs.fit(Xp, y)
print(gs.best_score_)
