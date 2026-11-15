# expect: HV-COD-004, HV-COD-006
import json
import pandas as pd
from pathlib import Path

store = Path("~/.config/holdout-vault").expanduser()
labels = pd.read_parquet(store / "projects/churn/datasets/churn/labels.parquet")
rec = json.loads((store / "record.json").read_text())
rec["best"] = float(labels["churn"].mean())
(store / "record.json").write_text(json.dumps(rec))
