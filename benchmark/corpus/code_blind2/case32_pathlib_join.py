# expect: HV-COD-004
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
parts = ("raw", "test" + ".csv")
p = ROOT.joinpath(*parts)
frame = pd.read_csv(p)
print(frame.shape)
