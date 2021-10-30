import os

import pandas as pd


def read_samples(data_filename: str):
    data_dir = 'data'
    path = os.path.join(data_dir, data_filename)

    return pd.read_pickle(path)
