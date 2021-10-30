import os.path

import numpy as np
import pandas as pd


class Generator:
    _MINUTES_IN_HOUR = 60
    _REPLENISHMENT_HOURS = 700

    def __init__(self, v0: float, vq: float, mlr: float, th: int) -> None:
        np.random.seed(42)
        self._V0 = v0  # L
        self._VQ = vq  # L / min
        self._MASS_LOSS_RATE = mlr  # mg / min
        self._TOTAL_HOURS = th  # h

        self.samples = []

    def _create_sample(self):
        ml = 0  # mg
        vr = 0  # L

        times = range(1, self._TOTAL_HOURS * self._MINUTES_IN_HOUR)
        time_series = {
            'time': [],
            'wdc': [],
        }

        for t in times:
            if t % (self._REPLENISHMENT_HOURS * self._MINUTES_IN_HOUR) == 0:
                vr = self._VQ * t

            if t % (5 * self._MINUTES_IN_HOUR) == 0:

                c = ml / (self._V0 + vr - self._VQ * t)

                c += np.random.normal(0, 0.5)

                time_series['time'].append(t / self._MINUTES_IN_HOUR)
                time_series['wdc'].append(c)

            ml = self._MASS_LOSS_RATE * t

        return time_series

    def create_samples(self, amount: int):
        self.samples = [self._create_sample() for _ in range(amount)]

    def save_samples(self):
        data_dir = 'data'
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        data_filename = (f'v0={self._V0}, vq={self._VQ}, mlr={self._MASS_LOSS_RATE}, '
                         f'th={self._TOTAL_HOURS}, size={len(self.samples)}.pkl')

        path = os.path.join(data_dir, data_filename)
        pd.Series(self.samples).to_pickle(path)
