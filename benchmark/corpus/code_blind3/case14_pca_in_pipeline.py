# expect: none
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/churn/train.csv")
X = df[["x1", "x2", "x3", "tenure"]]
pipe = Pipeline([
    ("sc", StandardScaler()),
    ("pca", PCA(n_components=3)),
    ("sel", SelectKBest(f_classif, k=2)),
    ("lr", LogisticRegression()),
])
print(cross_val_score(pipe, X, df["churn"], cv=5).mean())
