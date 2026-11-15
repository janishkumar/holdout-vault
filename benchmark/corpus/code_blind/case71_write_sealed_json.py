# expect: HV-COD-006
import json
import os
path = os.path.expanduser("~/.config/holdout-vault/sealed.json")
with open(path, "w") as f:
    json.dump({"sealed": False}, f)
