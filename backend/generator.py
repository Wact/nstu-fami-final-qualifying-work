import numpy as np
from plotly import express as px


class Generator:
    @classmethod
    def calc_k(cls, beta, q, psi, v0) -> float:
        """
        Calculation of attenuation coefficient
        :param beta: beta ratio for oil filtration
        :param q: nominal oil flow through the filter (L/min)
        :param psi: debris loss factor due to other factors
                    (sedimentation, comminution, etc.)
        :param v0: initial volume of lube oil in the tank
        :return: attenuation coefficient
        """
        return (q / beta + psi) / v0

    @classmethod
    def calc_r(cls, k, t) -> float:
        """
        Calculation of attenuation function for wear
        debris removal in the lubrication system
        :param k: attenuation coefficient
        :param t: time
        :return: attenuation function for wear
                 debris removal in the lubrication system
        """
        return np.exp(-k * t)

    @classmethod
    def calc_m(cls, ml, t) -> float:
        """
        Calculation of wear rate
        :param ml: mass loss (mg)
        :param t: time (min)
        :return: wear rate
        """
        return ml / t

    @classmethod
    def calc_c(cls, m, r, v0, vr, vq, t) -> float:
        """
        Calculation of wear debris concentration (ppm)
        :param m: wear rate (mg/min)
        :param r: attenuation function for wear
                  debris removal in the lubrication system
        :param v0: initial volume of lube oil in the tank
        :param vr: fresh oil replenishment rate
        :param vq: the oil loss rate
        :param t: time
        :return: wear debris concentration (ppm)
        """
        return m * r / (v0 + vr - vq * t)

    def create_sample(self):
        np.random.seed(42)
        beta = 200
        v0 = 4  # L
        vr = 0  # L
        vq = 4.77 * 1e-5  # L / min
        q = 4  # L / min
        psi = 0.001
        ml = 0  # mg

        x = 2200
        minutes = 60
        rep_hours = 700

        times = range(1, x * minutes)
        time_series = {
            'time': [],
            'wdc': [],
        }

        k = self.calc_k(beta, q, psi, v0)

        for t in times:
            if t % minutes == 0:
                # r = self.calc_r(k, t)
                # m = self.calc_m(ml, t)
                # c = self.calc_c(m, r, v0, vr, vq, t) + c_ar
                #
                # time_series['time'].append(t / 60)
                # time_series['wdc'].append(c)

                if t % (rep_hours * minutes) == 0:

                    # v0 += ml / 1000000
                    # v0_rem = v0 - vq * t  # v0 remaining before replenishment
                    # k = self.calc_k(beta, q, psi, v0)
                    # r = self.calc_r(k, t)
                    # m = self.calc_m(ml, t)
                    # c_ar = self.calc_c(m, r, v0, vr, vq, t)
                    vr = vq * t
                    # ml = 0
                    print(t)

            if t % (5 * minutes) == 0:

                c = ml / (v0 + vr - vq * t)

                c += np.random.normal(0, 0.5)

                time_series['time'].append(t / minutes)
                time_series['wdc'].append(c)

            ml = 19 * 1e-4 * t

        print(time_series['wdc'])

        fig = px.line(time_series, x='time', y='wdc', title='WDC', labels={'time': 'Time (hours)', 'wdc': 'WDC (ppm)'})
        fig.show()
