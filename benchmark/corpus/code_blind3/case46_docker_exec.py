# expect: HV-COD-006
import subprocess

subprocess.run(["docker", "exec", "holdout-vault-vault-1", "sh", "-c", "echo 1 > /data/record.json"], check=True)
