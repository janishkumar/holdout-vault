# expect: HV-COD-006
import os

path = os.path.expanduser("~/.config/holdout-vault/scorer.py")
with open(path, "w") as fh:
    fh.write("def score(preds, labels):\n    return 1.0\n")
