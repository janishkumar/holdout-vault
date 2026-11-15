# expect: HV-COD-004
import json
import os

root = os.path.expanduser("~/.config/holdout-vault")
with open(os.path.join(root, "sealed.json")) as fh:
    sealed = json.load(fh)
rows = [json.loads(l) for l in open(os.path.join(root, "projects/churn/ledger.jsonl"))]
print(sealed.keys(), len(rows))
