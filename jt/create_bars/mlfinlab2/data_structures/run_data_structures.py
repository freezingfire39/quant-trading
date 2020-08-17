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
        # cumulative buy and sell imbalances for a particular run calculation (theta_t in Prado book)
        cum_theta_buy = np.float(cache[-1].cum_theta_buy)
        cum_theta_sell = np.float(cache[-1].cum_theta_sell)
        # expected number of ticks extracted from prev bars
        exp_num_ticks = np.float(cache[-1].exp_num_ticks)
        # array of latest imbalances
        imbalance_array = cache[-1].imbalance_array
    else:
        # Reset counters
        cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0, 0
        high_price, low_price = -np.inf, np.inf
        exp_num_ticks, imbalance_array = exp_num_ticks_init, {
            'buy': [], 'sell': []}  # in run bars we need to track both buy and sell imbalance

    return cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks, imbalance_array


def _extract_bars(data, metric, exp_num_ticks_init=100000, num_prev_bars=3, num_ticks_ewma_window=20,
                  cache=None, flag=False, num_ticks_bar=None):
    """
    For loop which compiles the various imbalance bars: dollar, volume, or tick.

    :param data: Contains 3 columns - date_time, price, and volume.
    :param metric: dollar_imbalance, volume_imbalance or tick_imbalance
    :param exp_num_ticks_init: initial guess of number of ticks in imbalance bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param cache: contains information from the previous batch that is relevant in this batch.
    :param flag: A flag which signals to use the cache.
    :param num_ticks_bar: Expected number of ticks per bar used to estimate the next bar
    :param prev_tick_rule: Previous tick rule (if price_diff == 0 => use previous tick rule)
    :return: The financial data structure with the cache of short term history.
    """
    cache_tup = namedtuple('CacheData', ['date_time', 'price', 'high', 'low', 'tick_rule', 'cum_volume', 'cum_dollar_value',
                                         'cum_ticks', 'cum_theta_buy', 'cum_theta_sell', 'exp_num_ticks', 'imbalance_array'])
    if cache is None:
        cache = []
        prev_tick_rule = 0  # set the first tick rule with 0
        num_ticks_bar = []  # array of number of ticks from previous bars

    list_bars = []
    cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks, imbalance_array = _get_updated_counters(
        cache, flag, exp_num_ticks_init)

    # Iterate over rows
    for row in data.values:
        # Set variables
        date_time = row[0]
        price = np.float(row[1])
        volume = row[2]

        # Calculations
        cum_ticks += 1
        dollar_value = price * volume
        cum_dollar_value = cum_dollar_value + dollar_value
        cum_volume = cum_volume + volume

        # Imbalance calculations
        try:
            tick_diff = price - cache[-1].price
            prev_tick_rule = cache[-1].tick_rule
        except IndexError:
            tick_diff = 0

        tick_rule = np.sign(tick_diff) if tick_diff != 0 else prev_tick_rule

        if metric == 'tick_run':
            imbalance = tick_rule
        elif metric == 'dollar_run':
            imbalance = tick_rule * volume * price
        elif metric == 'volume_run':
            imbalance = tick_rule * volume

        if imbalance > 0:
            imbalance_array['buy'].append(imbalance)
            # set zero to keep buy and sell arrays synced
            imbalance_array['sell'].append(0)
            cum_theta_buy += imbalance
        elif imbalance < 0:
            imbalance_array['sell'].append(abs(imbalance))
            imbalance_array['buy'].append(0)
            cum_theta_sell += abs(imbalance)

        if len(imbalance_array['buy']) < exp_num_ticks:
            # waiting for array to fill for ewma
            exp_buy_proportion, exp_sell_proportion = np.nan, np.nan
        else:
            # expected imbalance per tick
            ewma_window = int(exp_num_ticks * num_prev_bars)
            buy_sample = np.array(
                imbalance_array['buy'][-ewma_window:], dtype=float)
            sell_sample = np.array(
                imbalance_array['sell'][-ewma_window:], dtype=float)
            buy_and_sell_imb = sum(buy_sample) + sum(sell_sample)
            exp_buy_proportion = ewma(
                buy_sample, window=ewma_window)[-1] / buy_and_sell_imb
            exp_sell_proportion = ewma(
                sell_sample, window=ewma_window)[-1] / buy_and_sell_imb

        # Check min max
        if price > high_price:
            high_price = price
        if price <= low_price:
            low_price = price

        # Update cache
        cache_data = cache_tup(date_time, price, high_price, low_price, tick_rule, cum_volume, cum_dollar_value,
                               cum_ticks, cum_theta_buy, cum_theta_sell, exp_num_ticks, imbalance_array)
        cache.append(cache_data)

        # Check expression for possible bar generation
        if max(cum_theta_buy, cum_theta_sell) > exp_num_ticks * max(exp_buy_proportion, exp_sell_proportion):   # pylint: disable=eval-used
            # Create bars
            open_price = cache[0].price
            low_price = min(low_price, open_price)
            close_price = price
            num_ticks_bar.append(cum_ticks)
            expected_num_ticks_bar = ewma(
                np.array(num_ticks_bar[-num_ticks_ewma_window:], dtype=float), num_ticks_ewma_window)[-1]  # expected number of ticks based on formed bars
            # Update bars & Reset counters
            list_bars.append([date_time, open_price, high_price, low_price, close_price,
                              cum_volume, cum_dollar_value, cum_ticks])
            cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0, 0
            high_price, low_price = -np.inf, np.inf
            exp_num_ticks = expected_num_ticks_bar
            cache = []

        # Update cache after bar generation (exp_num_ticks was changed after bar generation)
        cache_data = cache_tup(date_time, price, high_price, low_price, tick_rule, cum_volume, cum_dollar_value,
                               cum_ticks, cum_theta_buy, cum_theta_sell, exp_num_ticks, imbalance_array)
        cache.append(cache_data)
    return list_bars, cache, num_ticks_bar


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
            data=batch, metric=metric, exp_num_ticks_init=exp_num_ticks_init, num_prev_bars=num_prev_bars,
            num_ticks_ewma_window=num_ticks_ewma_window, cache=cache, flag=flag,
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


def get_dollar_run_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
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
    return _batch_run(file_path=file_path, metric='dollar_run', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)


def get_volume_run_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the volume imbalace bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    return _batch_run(file_path=file_path, metric='volume_run', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)


def get_tick_run_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the tick imbalace bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    return _batch_run(file_path=file_path, metric='tick_run', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)
