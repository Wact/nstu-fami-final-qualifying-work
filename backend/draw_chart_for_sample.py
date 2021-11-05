from plotly import express as px

from utils import read_samples


def draw_chart_for_sample() -> None:
    data_filename = 'v0=4, vq=4.77e-05, mlr=0.0019, th=2200, size=2.pkl'

    samples, _ = read_samples(data_filename, False)
    number = 0

    fig = px.line(
        samples[number], x='time', y='wdc', title='WDC',
        labels={'time': 'Time (hours)', 'wdc': 'WDC (ppm)'},
    )
    fig.show()


if __name__ == '__main__':
    draw_chart_for_sample()
