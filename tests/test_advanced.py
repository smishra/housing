# -*- coding: utf-8 -*-

from .context import housing

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_load_housing_data_no_id(self):
        data = housing.load_housing_data(data_path='./datasets/housing', to_create_id=False)
        assert len(data) > 0
        assert 'id' not in data

    def test_load_housing_data_with_id(self):
        data = housing.load_housing_data(data_path='./datasets/housing')
        assert len(data) > 0
        assert 'id' in data
        missing = data.isnull().sum()
        print("missing values\n", missing)
        assert missing.sum() > 0
        # print('shape: {}'.format(data.shape))
        # print('head: {}'.format(data.head()))
        # print('describe: {}'.format(data.describe()))

    def test_replace_na_by_median(self):
        data = housing.load_housing_data(data_path='./datasets/housing')
        print('original shape: {}'.format(data.shape))
        missing = data.isnull().sum()
        assert missing.sum() > 0
        replaced = housing.replace_na_by_median(data)
        print('shape after replace: {}'.format(replaced.shape))
        assert 'ocean_proximity' in replaced

    def test_create_income_category(self):
        data = housing.load_housing_data(data_path='./datasets/housing')
        assert 'income_cat' not in data
        housing.create_income_category_inplace(data)
        assert 'income_cat' in data
        assert data['income_cat'].value_counts().shape[0] == 5

    def start_collection(self):
        pass


if __name__ == '__main__':
    unittest.main()
