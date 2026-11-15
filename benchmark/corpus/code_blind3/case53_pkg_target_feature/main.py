import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from features import build

def main():
    df = pd.read_csv("data/churn/train.csv")
    m = RandomForestClassifier(random_state=0).fit(build(df), df["churn"])
    print(m.n_features_in_)

if __name__ == "__main__":
    main()
