import os
import tarfile
from six.moves import urllib
import pandas as pd

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_DATA_PATH = '../datasets/housing'

HOUSING_URL = DOWNLOAD_ROOT + HOUSING_DATA_PATH + '/housing.tgz'


def download_housing_data(housing_url=HOUSING_URL, housing_data_path=HOUSING_DATA_PATH):
    if not os.path.isdir(housing_data_path): os.makedirs(housing_data_path)
    tgz_path = os.path.join(housing_data_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_data_path)
    housing_tgz.close()


def load_housing_data(housing_data_path=HOUSING_DATA_PATH):
    csv_path = os.path.join(housing_data_path, 'housing.csv')
    return pd.read_csv(csv_path)
