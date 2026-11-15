# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

from sklearn.decomposition import PCA
df = pd.read_csv("data/churn/train.csv")
X, y = df[["x1", "x2", "x3", "tenure"]], df["churn"]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=0)
pca = PCA(n_components=2).fit(X_tr)
clf = LogisticRegression().fit(pca.transform(X_tr), y_tr)
print(clf.score(pca.transform(X_va), y_va))
