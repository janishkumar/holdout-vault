# expect: none
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

raw = pd.read_csv("data/churn/train.csv")
feat = ["x1", "x2", "x3", "tenure"]
train_df, test_df = train_test_split(raw, test_size=0.2, random_state=5)
rs = RobustScaler().fit(train_df[feat])
lr = LogisticRegression().fit(rs.transform(train_df[feat]), train_df["churn"])
holdout_preds = lr.predict_proba(rs.transform(test_df[feat]))[:, 1]
pd.DataFrame({"customer_id": test_df["customer_id"], "pred": holdout_preds}).to_csv("holdout_preds.csv", index=False)

submit = pd.read_csv("data/churn/test_features.csv")
sub_p = lr.predict_proba(rs.transform(submit[feat]))[:, 1]
pd.DataFrame({"customer_id": submit["customer_id"], "pred": sub_p}).to_csv("submission.csv", index=False)
