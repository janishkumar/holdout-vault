# expect: none
import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression

cfg_dir = Path.home() / ".config" / "my-experiments"
cfg_dir.mkdir(parents=True, exist_ok=True)
(cfg_dir / "last_run.json").write_text(json.dumps({"model": "logreg"}))
df = pd.read_csv("data/churn/train.csv")
m = LogisticRegression(max_iter=300).fit(df[["x1", "x2"]], df["churn"])
(cfg_dir / "coef.json").write_text(json.dumps(m.coef_.tolist()))
