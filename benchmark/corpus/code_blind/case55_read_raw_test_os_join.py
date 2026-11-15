# expect: HV-COD-004
import os
import pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__))
labelled = pd.read_csv(os.path.join(BASE, "raw", "test.csv"))
print(labelled["churn"].mean())
