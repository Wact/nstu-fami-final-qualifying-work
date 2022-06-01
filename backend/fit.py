import argparse
import logging
import os
import pickle
from math import sqrt

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn import svm, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from constants import Constants
from utils import read_samples, valid_algorithm, valid_svm_kernel, positive_int, get_data_filename_attribute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_and_save_model(model, x_test: pd.DataFrame, y_test: pd.Series, algorithm: str, size: int,
                            ts: str, rh: str, svm_kernel: str = ''):
    y_test_hat = model.predict(x_test)

    mae = metrics.mean_absolute_error(y_test, y_test_hat)
    rmse = sqrt(metrics.mean_squared_error(y_test, y_test_hat))
    r2 = metrics.r2_score(y_test, y_test_hat)

    model_name = f'{algorithm}-{svm_kernel}' if algorithm == 'svm' else algorithm

    logger.info(f'The MAE of {model_name} model is {mae}.')
    logger.info(f'The RMSE of {model_name} model is {rmse}.')
    logger.info(f'The R^2 score of {model_name} model is {r2}.')

    if not os.path.isdir(Constants.MODELS_DIR):
        os.mkdir(Constants.MODELS_DIR)

    model_filename = f'{model_name}, samples={size}, ts={ts}, rh={rh}, mae={mae:.3f}, rmse={rmse:.3f}, r2={r2:.3f}'
    model_path = os.path.join(Constants.MODELS_DIR, model_filename)

    if algorithm == 'catboost':
        model_filename += '.cbm'
        model_path += '.cbm'
        model.save_model(model_path)
    elif algorithm in ['linear', 'svm']:
        model_filename += '.pkl'
        model_path += '.pkl'
        pickle.dump(model, open(model_path, 'wb'))

    logger.info(f'Model filename is "{model_filename}".')


def fit() -> None:
    parser = argparse.ArgumentParser(description='Fit machine learning model.')

    # чтение параметра 'имя файла с подготовленными данными'
    parser.add_argument('-f', '--filename', required=True, dest='data_filename',
                        help='filename of pickle file with samples in \'data\\prepared_data\' folder')

    # чтение параметра 'алгоритм машинного обучения'
    parser.add_argument('-a', '--algorithm', required=True, dest='algorithm', type=valid_algorithm,
                        help='machine learning algorithm')

    # чтение параметра 'вид ядра в методе опорных векторов'
    parser.add_argument('-sk', '--svm_kernel', required=False, dest='svm_kernel', type=valid_svm_kernel, default='poly',
                        help='support vector machine kernel')

    # чтение параметра 'количество семплов для обучения'
    parser.add_argument('-sa', '--samples_amount', required=False, dest='samples_amount', type=positive_int, default=0,
                        help='amount of samples for training')

    args = parser.parse_args()
    data_filename = args.data_filename
    algorithm = args.algorithm
    svm_kernel = args.svm_kernel
    samples_amount = args.samples_amount

    rh = get_data_filename_attribute(data_filename, 'rh')
    ts = get_data_filename_attribute(data_filename, 'ts')

    # загрузка подготовленных данных
    df, size = read_samples(data_filename, is_prepared=True)

    # ограничение данных для обучения
    if samples_amount and samples_amount < size:
        df = df.iloc[:samples_amount]
        size = samples_amount

    # рассматриваемые признаки
    feature_columns = ['wdc_1']

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
        evaluate_and_save_model(model, x_test, y_test, algorithm, size, ts, rh)

    elif algorithm in ['linear', 'svm']:
        # обучение модели
        if algorithm == 'linear':
            model = LinearRegression()
        else:
            svm_params = {'kernel': svm_kernel}

            if svm_kernel == 'poly':
                svm_params['degree'] = 2

            model = svm.SVR(**svm_params)

            # обучающая выборка уменьшена до 100, потому что метод опорных векторов требует много памяти
            # при слишком большом количестве семплов
            x_train, y_train = x_train.iloc[:100], y_train.iloc[:100]

        model.fit(x_train, y_train)

        # сохранение модели
        evaluate_and_save_model(model, x_test, y_test, algorithm, size, ts, rh, svm_kernel)


if __name__ == '__main__':
    fit()
