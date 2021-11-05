import logging
import os

from catboost import CatBoostRegressor, Pool, cv

from constants import Constants
from utils import read_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit() -> None:
    # путь до файла с подготовленными данными для обучения
    data_filename = 'v0=4, vq=4.77e-05, mlr=0.0019, th=2200, samples=362275.pkl'

    # загрузка подготовленных данных
    df, size = read_samples(data_filename, is_prepared=True)

    # разделение данных на признаки и отклик
    train_pool = Pool(df.drop('target', axis=1), df['target'])

    # параметры обучаемой модели
    params = {
        'iterations': 100,
        'depth': 6,
        'loss_function': 'RMSE',
        'verbose': False,
    }

    # результаты кросс-валидации
    scores = cv(train_pool, params)

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

    #
    Constants.MODELS_DIR = 'models'
    if not os.path.isdir(Constants.MODELS_DIR):
        os.mkdir(Constants.MODELS_DIR)

    model_filename = f'cb_reg, samples={size}.cbm'

    model_path = os.path.join(Constants.MODELS_DIR, model_filename)

    model.save_model(model_path)


if __name__ == '__main__':
    fit()
