import argparse
import logging
import os

from catboost import CatBoostRegressor, Pool, cv

from constants import Constants
from utils import read_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit() -> None:
    parser = argparse.ArgumentParser(description='Fit machine learning model.')

    # чтение параметра 'имя файла с подготовленными данными'
    parser.add_argument('-f', '--filename', required=True, dest='data_filename',
                        help='filename of pickle file with samples in \'data\\prepared_data\' folder')

    data_filename = parser.parse_args().data_filename

    # загрузка подготовленных данных
    df, size = read_samples(data_filename, is_prepared=True)

    # разделение данных на признаки и отклик
    train_pool = Pool(df[['wdc_23', 'num_refill']], df['target'])

    # параметры обучаемой модели
    params = {
        'iterations': 500,
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

    model_filename = f'cb_reg, samples={size}.cbm'

    model_path = os.path.join(Constants.MODELS_DIR, model_filename)

    model.save_model(model_path)

    logger.info(f'Model filename is "{model_filename}".')


if __name__ == '__main__':
    fit()
