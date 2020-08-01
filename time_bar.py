import numpy as np
import pandas as pd
from scipy import stats

tick_data = pd.read_csv('AMD_dollar_all.csv', parse_dates=['date_time']).set_index('date_time')

timeframe = '1D'

ohlc_data = tick_data['close'].resample(timeframe).ohlc()

ohlc_data['volume'] = tick_data['cum_vol'].resample(timeframe).sum()

print (ohlc_data)

ohlc_data.to_csv('AMD_time.csv', index = 'date_time', header=True)
