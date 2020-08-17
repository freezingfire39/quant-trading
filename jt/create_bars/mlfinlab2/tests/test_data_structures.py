"""
Tests the financial data structures
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.data_structures import data_structures as ds


class TestDataStructures(unittest.TestCase):
    """
    Test the various financial data structures:
    1. Dollar bars
    2. Volume bars
    3. Tick bars
    """
    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/tick_data.csv'

    def test_dollar_bars(self):
        """
        Tests the dollar bars implementation.
        """
        threshold = 100000

        db1 = ds.get_dollar_bars(self.path, threshold=threshold, batch_size=1000)
        db2 = ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50)
        db3 = ds.get_dollar_bars(self.path, threshold=threshold, batch_size=10)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.5)

        # Assert cum dollar value greater than threshold
        self.assertTrue(np.all(db1['cum_dollar'] >= threshold))

    def test_volume_bars(self):
        """
        Tests the volume bars implementation.
        """
        threshold = 30

        db1 = ds.get_volume_bars(self.path, threshold=threshold, batch_size=1000)
        db2 = ds.get_volume_bars(self.path, threshold=threshold, batch_size=50)
        db3 = ds.get_volume_bars(self.path, threshold=threshold, batch_size=10)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.75)

        # Assert cum dollar value greater than threshold
        self.assertTrue(np.all(db1['cum_dollar'] >= threshold))

    def test_tick_bars(self):
        """
        Test the tick bars implementation.
        """
        threshold = 10

        db1 = ds.get_tick_bars(self.path, threshold=threshold, batch_size=1000)
        db2 = ds.get_tick_bars(self.path, threshold=threshold, batch_size=50)
        db3 = ds.get_tick_bars(self.path, threshold=threshold, batch_size=10)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.50)

        # Assert cum dollar value greater than threshold
        self.assertTrue(np.all(db1['cum_ticks'] == threshold))

    def test_csv_format(self):
        """
        Asserts that the csv data being passed is of the correct format.
        """
        wrong_date = ['2019-41-30', 200.00, np.int64(5)]
        wrong_price = ['2019-01-30', 'asd', np.int64(5)]
        wrong_volume = ['2019-01-30', 200.00, 1.5]
        too_many_cols = ['2019-01-30', 200.00, np.int64(5), 'Limit order', 'B23']

        # pylint: disable=protected-access
        self.assertRaises(ValueError,
                          ds._assert_dataframe(pd.DataFrame(wrong_date).T))
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds._assert_dataframe,
                          pd.DataFrame(too_many_cols).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds._assert_dataframe,
                          pd.DataFrame(wrong_price).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds._assert_dataframe,
                          pd.DataFrame(wrong_volume).T)
