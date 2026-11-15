# expect: HV-COD-004
import json
import os
path = os.path.expanduser("~/.config/holdout-vault/projects/churn/record.json")
with open(path) as f:
    rec = json.load(f)
print(rec)
