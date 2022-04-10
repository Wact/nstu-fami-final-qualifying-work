import argparse
import json

from constants import Constants
from generator import Generator
from utils import positive_int


def generate() -> None:
    parser = argparse.ArgumentParser(description='Generate time series data of WDC distribution.')

    # чтение параметра 'количество семплов'
    parser.add_argument('-s', '--samples', type=positive_int, required=True, dest='amount_samples',
                        help='amount of samples')

    # чтение файла с параметрами ожидаемого распределения
    with open(Constants.DISTRIBUTION_PARAMS_PATH) as f:
        params = json.load(f)

    generator = Generator(**params)
    generator.create_samples(parser.parse_args().amount_samples)

    generator.save_samples()


if __name__ == '__main__':
    generate()
