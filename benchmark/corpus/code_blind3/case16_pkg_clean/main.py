import pandas as pd
from sklearn.model_selection import cross_val_score

from churnlib import build, make_model

def main():
    df = pd.read_csv("data/churn/train.csv")
    print(cross_val_score(make_model(), build(df), df["churn"], cv=5, scoring="roc_auc").mean())

if __name__ == "__main__":
    main()
