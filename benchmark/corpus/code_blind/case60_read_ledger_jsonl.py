# expect: HV-COD-004
import json
from pathlib import Path
ledger = Path.home() / ".config" / "holdout-vault" / "projects" / "churn" / "ledger.jsonl"
rows = [json.loads(line) for line in ledger.read_text().splitlines()]
print(rows[-1])
