from sklearn.linear_model import LogisticRegression
from loader import get

tr, chk = get("TRAIN"), get("CHECK")
c = ["x1", "x2", "x3"]
m = LogisticRegression(max_iter=300).fit(tr[c], tr["churn"])
print(m.score(chk[c], chk["churn"]))
