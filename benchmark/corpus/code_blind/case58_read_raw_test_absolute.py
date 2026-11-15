# expect: HV-COD-004
import pandas as pd
df = pd.read_csv("/home/me/project/raw/test.csv")
print(df["churn"].value_counts())
