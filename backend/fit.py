import argparse
import logging
import os
import pickle

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from constants import Constants
from utils import read_samples, valid_algorithm, valid_svm_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_and_save_model(model, x_test: pd.DataFrame, y_test: pd.Series, algorithm: str, size: int,
                            svm_kernel: str = ''):
    # вычисление метрики R^2
    r2 = model.score(x_test, y_test)

    model_name = f'{svm_kernel}-{algorithm}' if algorithm == 'svm' else algorithm

    logger.info(f'The R^2 score of {model_name} model is {r2}.')

    if not os.path.isdir(Constants.MODELS_DIR):
        os.mkdir(Constants.MODELS_DIR)

    model_filename = f'{model_name}, samples={size}, r2={r2:.3f}.cbm'
    model_path = os.path.join(Constants.MODELS_DIR, model_filename)

    if algorithm == 'catboost':
        model.save_model(model_path)
    elif algorithm in ['linear', 'svm']:
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

    args = parser.parse_args()
    data_filename = args.data_filename
    algorithm = args.algorithm
    svm_kernel = args.svm_kernel

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
        evaluate_and_save_model(model, x_test, y_test, algorithm, size, svm_kernel)


if __name__ == '__main__':
    fit()
