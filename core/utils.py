import os
import tarfile
from six.moves import urllib
import pandas as pd
from sklearn.preprocessing import Imputer
import hashlib
import numpy as np


def download_remote_data(remote_url, local_data_path, local_filename):
    if not os.path.isdir(local_data_path): os.makedirs(local_data_path)
    tgz_path = os.path.join(local_data_path, local_filename)
    urllib.request.urlretrieve(remote_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=local_data_path)
    housing_tgz.close()


def csv_to_dataframe(data_path, csv_file_name):
    csv_path = os.path.join(data_path, csv_file_name)
    return pd.read_csv(csv_path)


def test_or_train(identifier, test_ratio=.2, hash_fn=hashlib.md5):
    """
    To verify if the identifier belongs to test or train set
    :param identifier: arg that's hash value is used
    :param test_ratio: the double value to determine the % of test case e.g., .2 means 20% to test 80% to train
    :param hash_fn: the hashing function, e.g, hashlib.md5, hashlib.sha256 etc
    :return: true if identifier belongs to test set
    """

    return hash_fn(np.int64(identifier)).digest[-1] < 256 * test_ratio


def split_test_train_by_id(data, id_column, test_ratio=.2, hash_fn=hashlib.md5):
    """
    Split the given data into test and train set determine by test_ratio
    :param data: pd.DataFrame data to be split
    :param id_column: the name of the id column (test_or_train function to use this column)
    :param test_ratio: the partition ratio in double, .2 means 20% test 80% train
    :param hash_fn: hash function to be used on the ids
    :return: test_set and train_set as pd.DataFrames
    """
    ids = data[id_column]
    test_or_train_ids = ids.apply(lambda id_: test_or_train(id_, test_ratio, hash_fn))
    return data[test_or_train_ids], data[~test_or_train_ids]


def replace_na_by_median(num_column_data):
    """
    Use median of the column value to replace na values
    :param num_column_data: pd.DataFrame of numerical type, median works only on numerical data
    :return: the pd.DataFrame (na replaced with median values)
    """
    imputer = Imputer(strategy='median')
    imputer.fit(num_column_data)
    print(imputer.statistics_)
    res_np_arr = imputer.transform(num_column_data)
    return pd.DataFrame(res_np_arr, columns=num_column_data.columns)


