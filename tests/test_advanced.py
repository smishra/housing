# -*- coding: utf-8 -*-

from .context import core

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_load_housing_data(self):
        data = core.utils.load_housing_data()
        assert len(data) >0
        print('shape: {}'.format(data.shape))
        print('head: {}'.format(data.head()))
        print('describe: {}'.format(data.describe()))



    def start_collection(self):
        pass

if __name__ == '__main__':
    unittest.main()
