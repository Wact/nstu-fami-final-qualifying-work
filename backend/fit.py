import argparse
import logging
import os
import pickle

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from constants import Constants
from utils import read_samples, valid_algorithm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_and_save_model(model, x_test: pd.DataFrame, y_test: pd.Series, algorithm: str, size: int):
    # вычисление метрики R^2
    r2 = model.score(x_test, y_test)

    logger.info(f'The R^2 score of {algorithm} model is {r2}.')

    if not os.path.isdir(Constants.MODELS_DIR):
        os.mkdir(Constants.MODELS_DIR)

    model_filename = f'{algorithm}, samples={size}, r2={r2:.3f}.cbm'
    model_path = os.path.join(Constants.MODELS_DIR, model_filename)

    if algorithm == 'catboost':
        model.save_model(model_path)
    elif algorithm == 'linear':
        pickle.dump(model, open(model_path, 'wb'))

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

    # разделение выборки на тренировочную и тестовую
    x_train, x_test, y_train, y_test = train_test_split(df[feature_columns], df['target'], test_size=0.2,
                                                        random_state=42)

    if algorithm == 'catboost':
        # параметры обучаемой модели
        params = {
            'iterations': 200,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False,
        }

        # инициализация модели
        model = CatBoostRegressor(**params)

        # обучение модели
        model.fit(x_train, y_train, eval_set=Pool(x_test, y_test))

        # сохранение модели
        evaluate_and_save_model(model, x_test, y_test, algorithm, size)

    elif algorithm == 'linear':
        # обучение модели
        model = LinearRegression()
        model.fit(x_train, y_train)

        # сохранение модели
        evaluate_and_save_model(model, x_test, y_test, algorithm, size)


if __name__ == '__main__':
    fit()
