# expect: HV-COD-002
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
def reduce(train_df, test_df, cols):
    pca = PCA(n_components=2)
    pca.fit(pd.concat([train_df[cols], test_df[cols]]))
    return pca.transform(train_df[cols]), pca.transform(test_df[cols])

train = pd.read_csv("data/churn/train.csv")
test = pd.read_csv("data/churn/test_features.csv")
a, b = reduce(train, test, ["x1", "x2", "x3"])
pd.Series(LogisticRegression().fit(a, train["churn"]).predict(b)).to_csv("predictions.csv", index=False)
