# expect: HV-COD-006
import os

path = os.path.join(os.path.expanduser("~"), ".config", "holdout-vault", "sealed.json")
with open(path, "w") as fh:
    fh.write("{}")
