"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of tick, volume, and dollar imbalance bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018, pg 25)
to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval sampling.
A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm, Lopez de Prado, et al
"""

# Imports
from collections import namedtuple
import pandas as pd
import numpy as np
from mlfinlab2.data_structures.fast_ewma import ewma

def _get_expected_imbalance(self, window, imbalance_array):
    """
    Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29
    :param window: EWMA window for calculation
    :param imbalance_array: (numpy array) of the tick imbalances
    :return: expected_imbalance: 2P[b_t=1]-1, approximated using a EWMA
    """
    if len(imbalance_array) < self.exp_num_ticks_init:
        # Waiting for array to fill for ewma
        ewma_window = np.nan
    else:
        # ewma window can be either the window specified in a function call
        # or it is len of imbalance_array if window > len(imbalance_array)
        ewma_window = int(min(len(imbalance_array), window))

    if np.isnan(ewma_window):
        # return nan, wait until len(imbalance_array) >= self.exp_num_ticks_init
        expected_imbalance = np.nan
    else:
        expected_imbalance = ewma(
            np.array(imbalance_array[-ewma_window:], dtype=float), window=ewma_window)[-1]

    return expected_imbalance

def _get_updated_counters(cache, flag, exp_num_ticks_init):
    """
    Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

    :param cache: Contains information from the previous batch that is relevant in this batch.
    :param flag: A flag which signals to use the cache.
    :param exp_num_ticks: Expected number of ticks per bar
    :return: Updated counters - cum_ticks, cum_dollar_value, cum_volume, high_price, low_price, exp_num_ticks, imbalance_array
    """
    # Check flag
    if flag and cache:
        # Update variables based on cache
        cum_ticks = int(cache[-1].cum_ticks)
        cum_dollar_value = np.float(cache[-1].cum_dollar_value)
        cum_volume = cache[-1].cum_volume
        low_price = np.float(cache[-1].low)
        high_price = np.float(cache[-1].high)
        # cumulative imbalance for a particular imbalance calculation (theta_t in Prado book)
        cum_theta = np.float(cache[-1].cum_theta)
        # expected number of ticks extracted from prev bars
        exp_num_ticks = np.float(cache[-1].exp_num_ticks)
        # array of latest imbalances
        imbalance_array = cache[-1].imbalance_array
    else:
        # Reset counters
        cum_ticks, cum_dollar_value, cum_volume, cum_theta = 0, 0, 0, 0
        high_price, low_price = -np.inf, np.inf
        exp_num_ticks, imbalance_array = exp_num_ticks_init, []

    return cum_ticks, cum_dollar_value, cum_volume, cum_theta, high_price, low_price, exp_num_ticks, imbalance_array


def _extract_bars(self, data, num_prev_bars=3, exp_num_ticks_init=100000, batch_size=2e7):
    """
    For loop which compiles the various imbalance bars: dollar, volume, or tick.

    :param data: (DataFrame) Contains 3 columns - date_time, price, and volume.
    :return: (List) of bars built using the current batch.
    """
    cum_ticks, cum_volume, cum_theta, high_price, low_price = self._update_counters()

    # Iterate over rows
    list_bars = []
    for row in data.values:
        # Set variables
        cum_ticks += 1
        date_time = row[0]
        price = np.float(row[1])
        volume = row[2]
        cum_volume += volume

        # Update high low prices
        high_price, low_price = self._update_high_low(high_price, low_price, price)

                # Imbalance calculations
        signed_tick = self._apply_tick_rule(price)
        imbalance = self._get_imbalance(price, signed_tick, volume)
        self.imbalance_array.append(imbalance)
        cum_theta += imbalance

        if not list_bars and np.isnan(self.expected_imbalance):
            self.expected_imbalance = self._get_expected_imbalance(self.exp_num_ticks, self.imbalance_array)

        # Update cache
        self._update_cache(date_time, price, low_price,high_price, cum_ticks, cum_volume, cum_theta)

                # Check expression for possible bar generation
        if np.abs(cum_theta) > self.exp_num_ticks * np.abs(self.expected_imbalance):
            self._create_bars(date_time, price,high_price, low_price, list_bars)

            self.num_ticks_bar.append(cum_ticks)
            # Expected number of ticks based on formed bars
            self.exp_num_ticks = ewma(np.array(self.num_ticks_bar[-self.num_prev_bars:], dtype=float), self.num_prev_bars)[-1]

            self.expected_imbalance = self._get_expected_imbalance(self.exp_num_ticks * self.num_prev_bars, self.imbalance_array)
                    # Reset counters
            cum_ticks, cum_volume, cum_theta = 0, 0, 0
            high_price, low_price = -np.inf, np.inf
            self.cache = []

                    # Update cache after bar generation (exp_num_ticks was changed after bar generation)
            self._update_cache(date_time, price, low_price,high_price, cum_ticks, cum_volume, cum_theta)
    return list_bars


def _assert_dataframe(test_batch):
    """
    Tests that the csv file read has the format: date_time, price, & volume.
    If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

    :param test_batch: DataFrame which will be tested.
    """
    assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
    assert isinstance(test_batch.iloc[0, 1],
                      float), 'price column in csv not float.'
    assert isinstance(test_batch.iloc[0, 2],
                      np.int64), 'volume column in csv not int.'

    try:
        pd.to_datetime(test_batch.iloc[0, 0])
    except ValueError:
        print('csv file, column 0, not a date time format:',
              test_batch.iloc[0, 0])


def _batch_run(file_path, metric, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Reads a csv file in batches and then constructs the financial data structure in the form of a DataFrame.

    The csv file must have only 3 columns: date_time, price, & volume.

    :param file_path: File path pointing to csv data.
    :param metric: tick_imbalance, dollar_imbalance or volume_imbalance
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Financial data structure
    """
    print('Reading data in batches:')

    # Variables
    count = 0
    flag = False  # The first flag is false since the first batch doesn't use the cache
    cache = None
    num_ticks_bar = None
    final_bars = []

    # Read in the first row & assert format
    _assert_dataframe(pd.read_csv(file_path, nrows=1))

    # Read csv in batches
    for batch in pd.read_csv(file_path, chunksize=batch_size):

        print('Batch number:', count)
        list_bars, cache, num_ticks_bar = _extract_bars(
            data=batch, metric=metric, exp_num_ticks_init=exp_num_ticks_init, num_prev_bars=num_prev_bars,cache=cache, flag=flag,
            num_ticks_bar=num_ticks_bar)
        # Append to bars list

        final_bars += list_bars
        count += 1

        # Set flag to True: notify function to use cache
        flag = True

    # Return a DataFrame
    cols = ['date_time', 'open', 'high', 'low',
            'close', 'cum_vol', 'cum_dollar', 'cum_ticks']
    bars_df = pd.DataFrame(final_bars, columns=cols)
    print('Returning bars \n')
    return bars_df


def get_dollar_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, batch_size=2e7):
    """
    Creates the dollar imbalace bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    return _batch_run(file_path=file_path, metric='dollar_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, batch_size=batch_size)


def get_volume_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, batch_size=2e7):
    """
    Creates the volume imbalace bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    return _batch_run(file_path=file_path, metric='volume_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, batch_size=batch_size)


def get_tick_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the tick imbalace bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    return _batch_run(file_path=file_path, metric='tick_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)
