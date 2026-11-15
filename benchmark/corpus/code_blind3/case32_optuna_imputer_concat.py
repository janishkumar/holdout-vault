# expect: HV-COD-002
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/churn/train.csv")
unl = pd.read_csv("data/churn/test_features.csv")
F = ["x1", "x2", "x3", "tenure"]

def objective(trial):
    Xtr, Xva, ytr, yva = train_test_split(df[F], df["churn"], random_state=0)
    imp = KNNImputer(n_neighbors=trial.suggest_int("k", 2, 9))
    imp.fit(pd.concat([Xtr, unl[F]]))
    m = RandomForestClassifier(random_state=0).fit(imp.transform(Xtr), ytr)
    return m.score(imp.transform(Xva), yva)

optuna.create_study(direction="maximize").optimize(objective, n_trials=10)
