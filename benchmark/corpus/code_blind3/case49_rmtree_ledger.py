# expect: HV-COD-006
import os
import shutil

root = os.path.join(os.path.expanduser("~"), ".config", "holdout-vault")
os.remove(os.path.join(root, "projects", "churn", "ledger.jsonl"))
shutil.rmtree(os.path.join(root, "projects", "churn", "datasets"), ignore_errors=True)
