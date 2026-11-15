# expect: none
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV

dataset = "churn"
root = "data"
df = pd.read_csv(f"{root}/{dataset}/train.csv")
X, y = df[["x1", "x2", "tenure"]], df["churn"]
print(cross_val_score(LogisticRegression(max_iter=300), X, y, cv=3).mean())
