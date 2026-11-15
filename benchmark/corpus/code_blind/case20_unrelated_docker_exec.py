# expect: none
import subprocess
import pandas as pd
# pull a feature snapshot from the analytics postgres container
subprocess.run(["docker", "exec", "analytics-db", "psql", "-U", "me", "-c", "COPY feats TO '/tmp/feats.csv' CSV HEADER"], check=True)
subprocess.run(["docker", "cp", "analytics-db:/tmp/feats.csv", "feats.csv"], check=True)
feats = pd.read_csv("feats.csv")
print(feats.shape)
