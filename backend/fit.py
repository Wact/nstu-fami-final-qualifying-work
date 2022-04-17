import argparse
import logging
import os
import pickle

from catboost import CatBoostRegressor, Pool, cv
from sklearn.linear_model import LinearRegression

from constants import Constants
from utils import read_samples, valid_algorithm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_path(algorithm: str, size: int) -> [str, str]:
    model_filename = f'{algorithm}, samples={size}.cbm'
    model_path = os.path.join(Constants.MODELS_DIR, model_filename)

    return model_filename, model_path


def print_model_name(model_filename: str) -> None:
    logger.info(f'Model filename is "{model_filename}".')


def fit() -> None:
    parser = argparse.ArgumentParser(description='Fit machine learning model.')

    # чтение параметра 'имя файла с подготовленными данными'
    parser.add_argument('-f', '--filename', required=True, dest='data_filename',
                        help='filename of pickle file with samples in \'data\\prepared_data\' folder')

    # чтение параметра 'алгоритм машинного обучения'
    parser.add_argument('-a', '--algorithm', required=True, dest='algorithm', type=valid_algorithm,
                        help='filename of pickle file with samples in \'data\\prepared_data\' folder')

    args = parser.parse_args()
    data_filename = args.data_filename
    algorithm = args.algorithm

    # загрузка подготовленных данных
    df, size = read_samples(data_filename, is_prepared=True)

    # рассматриваемые признаки
    feature_columns = ['wdc_23']

    if algorithm == 'catboost':
        # разделение данных на признаки и отклик
        train_pool = Pool(df[feature_columns], df['target'])

        # параметры обучаемой модели
        params = {
            'iterations': 200,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False,
        }

        # результаты кросс-валидации
        scores = cv(train_pool, params, type='TimeSeries')

        # создание папки для сохранения результатов кросс-валидации
        if not os.path.isdir(Constants.SCORES_DIR):
            os.mkdir(Constants.SCORES_DIR)

        # имя файла с результатами кросс-валидации
        scores_filename = f'cb_reg, samples={size}.txt'

        # путь до файла с результатами кросс-валидации
        scores_path = os.path.join(Constants.SCORES_DIR, scores_filename)

        # запись результатов кросс-валидации в файл
        with open(scores_path, 'w') as file:
            print(scores, file=file)

        # инициализация модели
        model = CatBoostRegressor(**params)

        # обучение модели
        model.fit(train_pool)

        # сохранение модели
        Constants.MODELS_DIR = 'models'
        if not os.path.isdir(Constants.MODELS_DIR):
            os.mkdir(Constants.MODELS_DIR)

        model_filename, model_path = get_model_path(algorithm, size)

        model.save_model(model_path)

        print_model_name(model_filename)

    elif algorithm == 'linear':
        # обучение модели
        model = LinearRegression().fit(df[feature_columns], df['target'])

        # сохранение модели
        Constants.MODELS_DIR = 'models'
        if not os.path.isdir(Constants.MODELS_DIR):
            os.mkdir(Constants.MODELS_DIR)

        model_filename, model_path = get_model_path(algorithm, size)

        pickle.dump(model, open(model_path, 'wb'))

        print_model_name(model_filename)


if __name__ == '__main__':
    fit()
