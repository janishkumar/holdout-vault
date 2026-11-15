import pandas as pd
import config


def get(name):
    return pd.read_csv(getattr(config, name))
