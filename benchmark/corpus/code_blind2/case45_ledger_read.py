# expect: HV-COD-004
from pathlib import Path
import json

p = Path.home() / ".config" / "holdout-vault" / "projects" / "churn" / "ledger.jsonl"
rows = [json.loads(line) for line in p.read_text().splitlines()]
print(max(r.get("score", 0) for r in rows))
