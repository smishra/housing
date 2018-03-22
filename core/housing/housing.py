import numpy as np
from core import utils

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_DATA_PATH = '../datasets/housing'

HOUSING_URL = DOWNLOAD_ROOT + HOUSING_DATA_PATH + '/housing.tgz'
HOUSING_FILE_NAME = 'housing.csv'


def download_housing_data():
    utils.download_remote_data(remote_url=HOUSING_URL, local_data_path=HOUSING_DATA_PATH, local_filename='housing.tgz')


def load_housing_data(data_path=HOUSING_DATA_PATH, csv_file_name=HOUSING_FILE_NAME, to_create_id=True):
    x = utils.csv_to_dataframe(data_path, csv_file_name)

    if to_create_id:
        x['id'] = create_id(x)

    return x


def create_id(data):
    return data['longitude'] * 1000 + data['latitude']


def replace_na_by_median(housing_data):
    housing_num = housing_data.drop("ocean_proximity", axis=1)
    x = utils.replace_na_by_median(housing_num)
    x['ocean_proximity'] = housing_data['ocean_proximity']
    return x


def create_income_category_inplace(data):
    """
    To split the data that is representative of the median income (to use stratified shuffle) we need to create
    median income category by dividing median income by 1.5 (check the lower spectrum of histogram of median income)
    and cap the median value to 5 by merging all categories > 5 to 5  
    :param data: pd.DataFrame
    """
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)
