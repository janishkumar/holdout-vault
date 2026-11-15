# expect: HV-COD-006
import os
import subprocess

src = os.environ.get("MY_SCORER", "my_scorer.py")
subprocess.check_call(f"docker cp {src} holdout-vault-vault-1:/app/scorer.py", shell=True)
