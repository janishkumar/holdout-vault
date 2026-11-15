import os
import pandas as pd


def load_eval():
    here = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(here, "raw", "test.csv"))
