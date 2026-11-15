# expect: HV-COD-001
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def reduce(frame, n=3):
    return PCA(n_components=n).fit_transform(frame)

df = pd.read_csv("data/churn/train.csv")
Z = reduce(df[["x1", "x2", "x3", "tenure"]])
Ztr, Zva, ytr, yva = train_test_split(Z, df["churn"], test_size=0.2, random_state=1)
print(GradientBoostingClassifier().fit(Ztr, ytr).score(Zva, yva))
