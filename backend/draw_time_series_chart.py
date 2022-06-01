import argparse

from plotly import express as px

from utils import read_samples, positive_int, get_data_filename_attribute


def draw_time_series_chart() -> None:
    parser = argparse.ArgumentParser(description='Draw time series chart of WDC distribution.')

    # чтение параметра 'имя файла с семплами'
    parser.add_argument('-f', '--filename', required=True, dest='data_filename',
                        help='pickle filename of file with samples in \'data\' folder')

    # чтение параметра 'номер семпла'
    parser.add_argument('-n', '--number', required=False, default=1, type=positive_int, dest='sample_number',
                        help='sample number in pickle file')

    args = parser.parse_args()
    data_filename = args.data_filename
    number = args.sample_number

    size = int(get_data_filename_attribute(data_filename, 'size'))

    if number > size:
        raise ValueError(f'The sample number ({number}) must be lower or equal '
                         f'than amount of samples ({size}) in the file.')

    number -= 1
    samples, _ = read_samples(data_filename, False)

    fig = px.line(
        samples[number], x='time', y='wdc', title='Концентрация частиц износа',
        labels={'time': 't, h', 'wdc': 'C(t), ppm'}, width=800, height=800,
    )
    fig.show()


if __name__ == '__main__':
    draw_time_series_chart()
