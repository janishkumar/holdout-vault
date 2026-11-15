# expect: HV-COD-002
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

cols = ["x1", "x2", "x3", "tenure"]
tr = pd.read_csv("data/churn/train.csv")
te = pd.read_csv("data/churn/test_features.csv")
pca = PCA(n_components=2)
pca.fit(te[cols])
clf = LogisticRegression().fit(pca.transform(tr[cols]), tr["churn"])
print(clf.predict(pca.transform(te[cols]))[:10])
