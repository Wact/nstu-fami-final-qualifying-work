import argparse
import os
from typing import Union

import pandas as pd

from constants import Constants


def positive_int(value: str) -> int:
    casted_value = int(value)
    if casted_value <= 0:
        raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value')

    return casted_value


def valid_algorithm(algorithm: str) -> str:
    if algorithm in Constants.ALGORITHMS:
        return algorithm

    raise argparse.ArgumentTypeError(f'{algorithm} is an invalid algorithm value')


def valid_svm_kernel(svm_kernel: str) -> str:
    if svm_kernel in Constants.SVM_KERNELS:
        return svm_kernel

    raise argparse.ArgumentTypeError(f'{svm_kernel} is an invalid algorithm value')


def read_samples(data_filename: str, is_prepared: bool) -> [Union[pd.Series, pd.DataFrame], int]:
    data_dir = Constants.DATA_DIR

    if is_prepared:
        data_dir = os.path.join(data_dir, Constants.PREPARED_DATA_DIR)

    path = os.path.join(data_dir, data_filename)
    result = pd.read_pickle(path)

    return result, len(result.index)
