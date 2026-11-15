# expect: none
from dataclasses import dataclass, field
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


@dataclass
class Cfg:
    train: str = "data/churn/train.csv"
    target: str = "churn"
    num: list = field(default_factory=lambda: ["x1", "x2", "x3", "tenure"])
    cat: list = field(default_factory=lambda: ["plan", "colour", "region"])


cfg = Cfg()
df = pd.read_csv(cfg.train)
pre = ColumnTransformer([
    ("num", StandardScaler(), cfg.num),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cfg.cat),
])
pipe = Pipeline([("pre", pre), ("gbm", lgb.LGBMClassifier(n_estimators=150, verbose=-1))])
oof = cross_val_predict(pipe, df[cfg.num + cfg.cat], df[cfg.target], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
print(roc_auc_score(df[cfg.target], oof[:, 1]))
