import pandas as pd

def load_data(path: str, n_rows=None):
    df = pd.read_csv(path, nrows=n_rows)
    return df
