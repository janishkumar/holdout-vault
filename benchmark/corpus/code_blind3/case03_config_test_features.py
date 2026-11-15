# expect: none
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

CFG = json.loads('{"train": "data/churn/train.csv", "test": "data/churn/test_features.csv", "target": "churn", "id": "customer_id", "features": ["x1", "x2", "x3", "tenure"]}')

def load(cfg):
    return pd.read_csv(cfg["train"]), pd.read_csv(cfg["test"])

def main():
    tr, te = load(CFG)
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(tr[CFG["features"]], tr[CFG["target"]])
    out = te[[CFG["id"]]].assign(p=model.predict_proba(te[CFG["features"]])[:, 1])
    out.to_csv("preds.csv", index=False)

if __name__ == "__main__":
    main()
