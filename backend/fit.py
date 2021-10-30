from utils import read_samples


def fit():
    data_filename = 'v0=4, vq=4.77e-05, mlr=0.0019, th=2200, size=2.pkl'

    read_samples(data_filename)


if __name__ == '__main__':
    fit()
