# expect: none
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("data/churn/train.csv")
FEATS = ["x1", "x2", "x3", "tenure"]

def objective(trial):
    Xtr, Xva, ytr, yva = train_test_split(df[FEATS], df["churn"], test_size=0.25, random_state=trial.number)
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        GradientBoostingClassifier(
            learning_rate=trial.suggest_float("lr", 0.01, 0.3, log=True),
            n_estimators=trial.suggest_int("n", 50, 300),
        ),
    )
    pipe.fit(Xtr, ytr)
    return roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
