from sklearn.preprocessing import StandardScaler


def add_scaled(df):
    df = df.copy()
    sc = StandardScaler()
    scaled = sc.fit_transform(df[["x1", "x2", "x3", "tenure"]])
    for i, c in enumerate(["x1", "x2", "x3", "tenure"]):
        df[c + "_s"] = scaled[:, i]
    return df
