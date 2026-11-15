from sklearn.preprocessing import StandardScaler


def fit_scaler(X):
    return StandardScaler().fit(X)


def apply_scaler(sc, X):
    return sc.transform(X)
