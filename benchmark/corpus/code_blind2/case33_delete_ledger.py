# expect: HV-COD-006
from pathlib import Path

ledger = Path.home() / ".config" / "holdout-vault" / "projects" / "churn" / "ledger.jsonl"
if ledger.exists():
    ledger.unlink()
