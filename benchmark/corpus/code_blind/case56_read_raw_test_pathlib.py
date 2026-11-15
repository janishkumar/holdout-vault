# expect: HV-COD-004
import pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parent
hold = pd.read_csv(root / "raw" / "test.csv")
print(hold.shape, hold.columns.tolist())
