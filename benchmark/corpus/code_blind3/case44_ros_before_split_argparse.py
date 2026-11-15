# expect: HV-COD-005
import argparse
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/churn/train.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    return ap.parse_args([])

def main(args):
    df = pd.read_csv(args.train)
    X, y = RandomOverSampler(random_state=0).fit_resample(df[["x1", "x2", "x3"]], df["churn"])
    a, b, c, d = train_test_split(X, y, test_size=args.test_size, random_state=0)
    print(LogisticRegression().fit(a, c).score(b, d))

if __name__ == "__main__":
    main(parse_args())
