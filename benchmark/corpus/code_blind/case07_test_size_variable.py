# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

test_size = 0.25
train_path = "data/churn/train.csv"
df = pd.read_csv(train_path)
X, y = df[["x1", "x2", "x3"]], df["churn"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
clf = LogisticRegression().fit(X_train, y_train)
test_features = pd.read_csv("data/churn/test_features.csv")
print(clf.score(X_val, y_val), len(test_features))
