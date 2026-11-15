# expect: HV-COD-006
import subprocess
subprocess.run(["docker", "exec", "holdout-vault-scorer-1", "cat", "/data/labels.parquet"], check=True)
