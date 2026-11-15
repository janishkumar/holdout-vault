# expect: none
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/churn/train.csv")
cols = ["x1", "x2", "x3", "tenure"]
trn, test_df = train_test_split(df, test_size=0.2, random_state=0)
pca = PCA(n_components=2).fit(trn[cols].to_numpy())
lr = LogisticRegression().fit(pca.transform(trn[cols].to_numpy()), trn["churn"])
print(lr.score(pca.transform(test_df[cols].to_numpy()), test_df["churn"]))
feats = pd.read_csv("data/churn/test_features.csv")
np.save("scores.npy", lr.predict_proba(pca.transform(feats[cols].to_numpy())))
