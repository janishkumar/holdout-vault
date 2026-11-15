import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import load_test, load_train

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=1.0)
    return ap.parse_args([])

def main(args):
    tr, te = load_train(), load_test()
    cols = ["x1", "x2", "x3"]
    sc = StandardScaler().fit(tr[cols])
    clf = LogisticRegression(C=args.c).fit(sc.transform(tr[cols]), tr["churn"])
    te["p"] = clf.predict_proba(sc.transform(te[cols]))[:, 1]
    te[["customer_id", "p"]].to_csv("p.csv", index=False)

if __name__ == "__main__":
    main(parse_args())
