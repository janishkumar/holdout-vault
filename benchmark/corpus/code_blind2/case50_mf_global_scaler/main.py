from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from prep import load_and_scale, NUM

df, _ = load_and_scale("data/churn/train.csv")
tr, va = train_test_split(df, test_size=0.25, random_state=0)
g = GradientBoostingClassifier().fit(tr[NUM], tr["churn"])
print(g.score(va[NUM], va["churn"]))
