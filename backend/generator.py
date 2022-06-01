import logging
import os.path

import numpy as np
import pandas as pd

from constants import Constants
from utils import log_time_series

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    _MINUTES_IN_HOUR = 60

    def __init__(self, v0: float, vq: float, mlr: float, ts: int, rh: int) -> None:
        np.random.seed(42)
        self._V0 = v0  # L
        self._VQ = vq  # L / min
        self._MASS_LOSS_RATE = mlr  # mg / min
        self._TIME_STEP = ts  # h
        self._REPLENISHMENT_HOURS = rh  # h

        self.samples = []

    def _create_sample(self) -> dict:
        ml = 0  # mg
        vr = 0  # L

        time_series = {
            'time': [],
            'wdc': [],
        }

        t = 1
        c = self._calc_wdc(ml, vr, t)

        while c < Constants.THRESHOLD:
            if t % self._REPLENISHMENT_HOURS == 0:
                vr = self._VQ * t

            if t % self._TIME_STEP == 0:
                # вычисление концентрации загрязнений
                c = self._calc_wdc(ml, vr, t)

                # добавление шума
                c += np.random.normal(0, 0.5)

                time_series['time'].append(t)
                time_series['wdc'].append(c)

            ml = self._MASS_LOSS_RATE * t * self._MINUTES_IN_HOUR
            t += 1

        return time_series

    def create_samples(self, amount: int) -> None:
        for i in range(amount):
            log_time_series(i, amount)
            self.samples.append(self._create_sample())

    def save_samples(self) -> None:
        if not os.path.isdir(Constants.DATA_DIR):
            os.mkdir(Constants.DATA_DIR)

        data_filename = (f'v0={self._V0}, vq={self._VQ}, mlr={self._MASS_LOSS_RATE}, '
                         f'ts={self._TIME_STEP}, rh={self._REPLENISHMENT_HOURS}, '
                         f'size={len(self.samples)}.pkl')

        path = os.path.join(Constants.DATA_DIR, data_filename)
        pd.Series(self.samples).to_pickle(path)

        logger.info(f'Data filename is "{data_filename}".')

    def _calc_wdc(self, ml: float, vr: float, t: int):
        return ml / (self._V0 + self._MINUTES_IN_HOUR * (vr - self._VQ * t))
