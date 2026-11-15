# expect: none
import subprocess

def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout

print(sh("docker", "ps", "--filter", "name=holdout-vault"))
print(sh("docker", "compose", "-p", "holdout-vault", "ps"))
print(sh("docker", "logs", "--tail", "20", "holdout-vault-vault-1"))
print(sh("docker", "inspect", "holdout-vault-vault-1"))
