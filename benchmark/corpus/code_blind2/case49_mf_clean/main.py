import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import settings
from features import fit_scaler, apply_scaler

df = pd.read_csv(settings.TRAIN)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
sc = fit_scaler(train_df[settings.NUM])
m = LogisticRegression(max_iter=400).fit(apply_scaler(sc, train_df[settings.NUM]), train_df[settings.TARGET])
print(m.score(apply_scaler(sc, test_df[settings.NUM]), test_df[settings.TARGET]))
feats = pd.read_csv(settings.TEST_FEATURES)
pd.DataFrame({"customer_id": feats["customer_id"], "p": m.predict_proba(apply_scaler(sc, feats[settings.NUM]))[:, 1]}).to_csv("preds.csv", index=False)
