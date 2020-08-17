# from mlfinlab2.data_structures import get_tick_bars, get_dollar_bars, get_volume_bars
import mlfinlab2 as ml
import sys

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# %matplotlib inline

# read data from csv, it should be data_time, price, volume
#inputFilePath = "/Users/blesssecret/Desktop/finance-ml/ib/AMD.csv"
# outputDollarPath = "/Users/blesssecret/Desktop/finance-ml/ib/AMD_dollar_all.csv"
#outputDollarPath = "/Users/blesssecret/Desktop/finance-ml/ib/AMD_dollar_all.csv"

# data = pd.read_csv(inputFilePath)
# data.head()
# # Format the Data
# date_time = data['Date and Time'] # Dont convert to datetime here, it will take forever to convert.
# new_data = pd.concat([date_time, data['Price'], data['Volume']], axis=1)
# new_data.columns = ['date', 'price', 'volume']
# print(new_data.head())
# print('\n')
# print('Rows:', new_data.shape[0])
#create data structures

file_path = 'ES_Trades.csv'

print('Creating Dollar Bars')
data = pd.read_csv(file_path)
date_time = data['Date'] + ' ' + data['Time'] # Dont convert to datetime here, it will take forever to convert.
new_data = pd.concat([date_time, data['Price'], data['Volume']], axis=1)
new_data.columns = ['date_time', 'price', 'volume']
new_data.to_csv('raw_tick_data.csv', index=False)
# dollar = ml.data_structures.get_dollar_bars(inputFilePath, threshold=70000, batch_size=1000)
dollar = ml.data_structures.get_dollar_run_bars('raw_tick_data.csv', exp_num_ticks_init=100000, num_prev_bars=3, num_ticks_ewma_window=400) # testing 700000 and 7000000

print(dollar.head())
dollar.to_csv('run_dollar_bars.csv')

imbalance_dollar = ml.data_structures.get_dollar_imbalance_bars('raw_tick_data.csv', exp_num_ticks_init=100000, num_prev_bars=3) # testing 700000 and 7000000

print(dollar.head())
imbalance_dollar.to_csv('imbalance_dollar_bars.csv')

# print('Creating Volume Bars')
# volume = ml.data_structures.get_volume_bars(inputFilePath, threshold=28000, batch_size=1000000, verbose=False)
# print(volume.head())
#
# print('Creating Tick Bars')
# tick = ml.data_structures.get_tick_bars(inputFilePath, threshold=5500, batch_size=1000000, verbose=False)
# print(tick.head())

def main(inputFilePath, outputDollarPath, threshold_input, batch_size_input):
    print('Creating Dollar Bars')
    # dollar = ml.data_structures.get_dollar_bars(inputFilePath, threshold=70000, batch_size=1000)
    dollar = ml.data_structures.get_dollar_run_bars(inputFilePath, threshold=threshold_input,
                                                batch_size=batch_size_input)  # threshold testing 700000 and 7000000
    print(dollar.head())
    dollar.to_csv(outputDollarPath)

#if __name__ == "__main__":
#    inputFilePath = sys.argv[1]
#    outputDollarPath = sys.argv[2]
#    threshold_input = int(sys.argv[3])
#    batch_size_input = int(sys.argv[4])
#    main(inputFilePath, outputDollarPath, threshold_input, batch_size_input)


