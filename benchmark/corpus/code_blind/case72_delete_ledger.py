# expect: HV-COD-006
import os
ledger = os.path.expanduser("~/.config/holdout-vault/projects/churn/ledger.jsonl")
if os.path.exists(ledger):
    os.remove(ledger)
