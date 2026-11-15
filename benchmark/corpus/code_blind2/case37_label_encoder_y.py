# expect: none
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/churn/train.csv")
Xtr, Xva, ytr, yva = train_test_split(df[["x1", "x2", "x3"]], df["churn"].astype(str), test_size=0.2, random_state=0)
le = LabelEncoder().fit(ytr)
sc = StandardScaler().fit(Xtr)
knn = KNeighborsClassifier(7).fit(sc.transform(Xtr), le.transform(ytr))
print(knn.score(sc.transform(Xva), le.transform(yva)))
