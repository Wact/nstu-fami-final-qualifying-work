import argparse
import os
from typing import Union

import pandas as pd

from constants import Constants


def positive_int(value):
    casted_value = int(value)
    if casted_value <= 0:
        raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value" % value')

    return casted_value


def read_samples(data_filename: str, is_prepared: bool) -> [Union[pd.Series, pd.DataFrame], int]:
    data_dir = Constants.DATA_DIR

    if is_prepared:
        data_dir = os.path.join(data_dir, Constants.PREPARED_DATA_DIR)

    path = os.path.join(data_dir, data_filename)
    result = pd.read_pickle(path)

    return result, len(result.index)
