# expect: HV-COD-004
import pandas as pd
root = "."
split = "test"
frame = pd.read_csv(f"{root}/raw/{split}.csv")
print(frame.describe())
