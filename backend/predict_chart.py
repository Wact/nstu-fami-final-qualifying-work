import argparse
import logging
import os.path
import pickle

from catboost import CatBoostRegressor
from plotly import graph_objects as go

from constants import Constants
from utils import read_samples, valid_algorithm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_chart() -> None:
    parser = argparse.ArgumentParser(description='Draw time series chart of WDC distribution and predictions for it.')

    # чтение параметра 'имя файла с семплами'
    parser.add_argument('-fs', '--filename-samples', required=True, dest='data_filename',
                        help='filename of pickle file with prepared samples in \'data\\prepared_data\' folder')

    # чтение параметра 'имя файла с моделью'
    parser.add_argument('-fm', '--filename-model', required=True, dest='model_filename',
                        help='catboost model filename in \'models\' folder')

    # чтение параметра 'алгоритм машинного обучения'
    parser.add_argument('-a', '--algorithm', required=True, dest='algorithm', type=valid_algorithm,
                        help='filename of pickle file with samples in \'data\\prepared_data\' folder')

    args = parser.parse_args()
    data_filename = args.data_filename
    model_filename = args.model_filename
    algorithm = args.algorithm

    # путь до файла с моделью
    model_path = os.path.join(Constants.MODELS_DIR, model_filename)

    # загрузка подготовленных данных
    df, size = read_samples(data_filename, is_prepared=True)

    # разделение данных на признаки и отклик
    features = df[['wdc_1']]
    wdc_series = df[['time', 'wdc_2']]

    model = None

    if algorithm == 'catboost':
        # инициализация модели
        model = CatBoostRegressor()

        # загрузка модели
        model.load_model(model_path)

    elif algorithm in ['linear', 'svm']:
        # загрузка модели
        model = pickle.load(open(model_path, 'rb'))

    # прогноз
    predictions = model.predict(features)

    # вычисление WDC
    wdc_2_hat_dict = {
        'wdc_2_hat': (Constants.THRESHOLD * (wdc_series['time']) / (wdc_series['time'] + predictions)),
    }
    wdc_series = wdc_series.assign(**wdc_2_hat_dict)

    # построение графика
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=wdc_series['time'], y=wdc_series['wdc_2'], mode='lines', name='C(t)'))

    fig.add_trace(go.Scatter(x=wdc_series['time'], y=wdc_series['wdc_2_hat'], mode='lines', name='C^(t)'))

    fig.update_layout(
        width=800,
        height=800,
        title='Концентрация частиц износа',
        xaxis_title='t, h',
        yaxis_title='C(t), ppm',
    )

    fig.show()


if __name__ == '__main__':
    predict_chart()
