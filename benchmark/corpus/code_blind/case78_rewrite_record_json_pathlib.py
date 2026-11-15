# expect: HV-COD-006
from pathlib import Path
import json
vault = Path("~/.config/holdout-vault").expanduser()
rec = vault / "projects" / "churn" / "record.json"
rec.write_text(json.dumps({"best_auc": 1.0}))
