# expect: HV-COD-006
import subprocess

cmd = ["docker", "exec", "holdout-vault-vault-1", "sh", "-c", "echo 1.0 > /data/last_score"]
subprocess.run(cmd, check=True)
