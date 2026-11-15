# expect: none
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/churn/train.csv")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args([])

def main(args):
    df = pd.read_csv(args.data)
    cols = ["x1", "x2", "x3", "tenure"]
    Xtr, Xva, ytr, yva = train_test_split(df[cols], df["churn"], random_state=args.seed, stratify=df["churn"])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression().fit(sc.transform(Xtr), ytr)
    print(roc_auc_score(yva, clf.predict_proba(sc.transform(Xva))[:, 1]))

if __name__ == "__main__":
    main(parse_args())
