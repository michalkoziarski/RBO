import os
import zipfile
import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from imblearn.datasets import make_imbalance

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
FOLDS_PATH = os.path.join(os.path.dirname(__file__), 'folds')


def download(url):
    name = url.split('/')[-1]
    download_path = os.path.join(DATA_PATH, name)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(download_path):
        urlretrieve(url, download_path)

    if not os.path.exists(download_path.replace('.zip', '.dat')):
        if name.endswith('.zip'):
            with zipfile.ZipFile(download_path) as zip:
                zip.extractall(DATA_PATH)
        else:
            raise Exception('Unrecognized file type.')


def encode(X, y, encode_features=True):
    y = preprocessing.LabelEncoder().fit(y).transform(y)

    if encode_features:
        encoded = []

        for i in range(X.shape[1]):
            try:
                float(X[0, i])
                encoded.append(X[:, i])
            except:
                encoded.append(preprocessing.LabelEncoder().fit_transform(X[:, i]))

        X = np.transpose(encoded)

    return X.astype(np.float32), y.astype(np.float32)


def partition(X, y):
    partitions = []

    for _ in range(5):
        folds = []
        skf = StratifiedKFold(n_splits=2, shuffle=True)

        for train_idx, test_idx in skf.split(X, y):
            folds.append([train_idx, test_idx])

        partitions.append(folds)

    return partitions


def make_folds(X, y, partitions, scale=True, noise_type=None, noise_level=0.0):
    folds = []

    for i in range(5):
        for j in range(2):
            train_idx, test_idx = partitions[i][j]
            train_set = [X[train_idx].copy(), y[train_idx].copy()]
            test_set = [X[test_idx].copy(), y[test_idx].copy()]

            if scale:
                scaler = MinMaxScaler().fit(train_set[0])
                train_set[0] = scaler.transform(train_set[0])
                test_set[0] = scaler.transform(test_set[0])

            if noise_type == 'class' and noise_level > 0.0:
                classes = np.unique(y)
                sizes = [sum(y == c) for c in classes]
                minority_class = classes[np.argmin(sizes)]
                majority_class = classes[np.argmax(sizes)]

                assert minority_class != majority_class

                for k in range(len(train_set[1])):
                    if train_set[1][k] == majority_class and np.random.rand() < noise_level:
                        train_set[1][k] = minority_class

            if noise_type == 'attribute' and noise_level > 0.0:
                maximum = np.max(train_set[0], axis=0)
                minimum = np.min(train_set[0], axis=0)

                for k in range(train_set[0].shape[1]):
                    train_set[0][:, k] += np.random.normal(loc=0.0, scale=noise_level * (maximum[k] - minimum[k]) / 5.0,
                                                           size=train_set[0].shape[0])

            folds.append([train_set, test_set])

    return folds


def load(name, url=None, encode_features=True, remove_metadata=True, scale=True, noise_type=None, noise_level=0.0):
    assert noise_type in [None, 'class', 'attribute']
    assert 0.0 <= noise_level <= 1.0

    file_name = '%s.dat' % name

    if url is not None:
        download(url)

    skiprows = 0

    if remove_metadata:
        with open(os.path.join(DATA_PATH, file_name)) as f:
            for line in f:
                if line.startswith('@'):
                    skiprows += 1
                else:
                    break

    df = pd.read_csv(os.path.join(DATA_PATH, file_name), header=None, skiprows=skiprows, skipinitialspace=True,
                     sep=' *, *', na_values='?', engine='python')

    matrix = df.dropna().as_matrix()

    X, y = matrix[:, :-1], matrix[:, -1]
    X, y = encode(X, y, encode_features)

    partitions_path = os.path.join(FOLDS_PATH, file_name.replace('.dat', '.folds.pickle'))

    if not os.path.exists(FOLDS_PATH):
        os.mkdir(FOLDS_PATH)

    if os.path.exists(partitions_path):
        partitions = pickle.load(open(partitions_path, 'rb'))
    else:
        partitions = partition(X, y)
        pickle.dump(partitions, open(partitions_path, 'wb'))

    return make_folds(X, y, partitions, scale, noise_type, noise_level)


def load_all(type=None):
    assert type in [None, 'preliminary', 'final']

    urls = []

    for current_type in ['preliminary', 'final']:
        if type in [None, current_type]:
            with open(os.path.join(os.path.dirname(__file__), 'data_urls', '%s.txt' % current_type)) as file:
                for line in file.readlines():
                    urls.append(line.rstrip())

    datasets = {}

    for url in urls:
        name = url.split('/')[-1].replace('.zip', '')
        datasets[name] = load(name, url)

    return datasets


def names(type=None):
    assert type in [None, 'preliminary', 'final']

    urls = []

    for current_type in ['preliminary', 'final']:
        if type in [None, current_type]:
            with open(os.path.join(os.path.dirname(__file__), 'data_urls', '%s.txt' % current_type)) as file:
                for line in file.readlines():
                    urls.append(line.rstrip())

    return [url.split('/')[-1].replace('.zip', '') for url in urls]


def make_synthetic(n_majority_samples, n_features, imbalance_ratio, scale=True, noise_type=None, noise_level=0.0):
    assert imbalance_ratio >= 1.0

    n_informative_redundant = np.min((2, int(0.1 * n_features)))

    X, y = make_classification(int(2 * n_majority_samples * 1.05), n_features,
                               n_informative=n_informative_redundant,
                               n_redundant=n_informative_redundant)
    X, y = make_imbalance(X, y, {0: n_majority_samples, 1: int(n_majority_samples / imbalance_ratio)})

    partitions = partition(X, y)

    return make_folds(X, y, partitions, scale, noise_type, noise_level)


if __name__ == '__main__':
    load_all()
