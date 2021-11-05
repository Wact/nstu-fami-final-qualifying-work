import logging
import os

import pandas as pd

from constants import Constants
from utils import read_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data():
    data_filename = 'v0=4, vq=4.77e-05, mlr=0.0019, th=2200, size=2.pkl'

    samples, size = read_samples(data_filename, is_prepared=False)

    columns = [f'wdc_{x}' for x in range(Constants.WINDOW_SIZE)] + ['time'] + ['target']

    d = {}
    global_index = 0

    # подготовка данных
    for i in range(size):
        logger.info(f'Time series #{i}')

        sample: dict = samples[i]
        wdc = pd.Series(sample['wdc'])
        threshold_index = wdc.index[wdc > Constants.THRESHOLD][0]
        threshold_hour = sample['time'][threshold_index]

        j = 0

        for j in range(threshold_index - Constants.WINDOW_SIZE):
            wdc_window = sample['wdc'][j:j + Constants.WINDOW_SIZE]
            time_first = sample['time'][j]
            time_last = sample['time'][j + Constants.WINDOW_SIZE - 1]
            target = threshold_hour - time_last
            d[global_index + j] = pd.Series(wdc_window + [time_first] + [target], index=columns)

        global_index += j + 1

    df = pd.DataFrame.from_dict(d, columns=columns, orient='index')

    prepared_data_dir = os.path.join(Constants.DATA_DIR, Constants.PREPARED_DATA_DIR)
    if not os.path.isdir(prepared_data_dir):
        os.mkdir(prepared_data_dir)

    data_filename = ', '.join((data_filename.split(', ')[:-1]) + [f'samples={global_index}.pkl'])

    path = os.path.join(prepared_data_dir, data_filename)
    df.to_pickle(path)


if __name__ == '__main__':
    prepare_data()
