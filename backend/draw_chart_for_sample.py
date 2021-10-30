import os

import pandas as pd
from plotly import express as px


def draw_chart_for_sample():
    data_dir = 'data'
    data_filename = 'v0=4, vq=4.77e-05, mlr=0.0019, th=2200, size=2.pkl'
    path = os.path.join(data_dir, data_filename)

    samples = pd.read_pickle(path)
    number = 0

    fig = px.line(
        samples[number], x='time', y='wdc', title='WDC',
        labels={'time': 'Time (hours)', 'wdc': 'WDC (ppm)'},
    )
    fig.show()


if __name__ == '__main__':
    draw_chart_for_sample()
