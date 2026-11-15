from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FEATURES

__all__ = ["make_model"]

def make_model():
    cat = ["plan", "region"]
    num = [c for c in FEATURES if c not in cat] + ["x12"]
    ct = ColumnTransformer([("o", OneHotEncoder(handle_unknown="ignore"), cat), ("s", StandardScaler(), num)])
    return Pipeline([("prep", ct), ("gb", GradientBoostingClassifier())])
