# expect: none
import subprocess

out = subprocess.run(["docker", "compose", "-p", "holdout-vault", "ps", "--format", "json"], capture_output=True, text=True)
print(out.stdout)
logs = subprocess.run(["docker", "logs", "--tail", "20", "holdout-vault-vault-1"], capture_output=True, text=True)
print(logs.stdout)
