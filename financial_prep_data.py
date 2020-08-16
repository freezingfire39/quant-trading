from sys import stdout
import numpy as np
import pandas as pd
from pandas_datareader import data
import json

# Reading data from external sources
import urllib as u
from urllib.request import urlopen

# Machine learning (preprocessing, models, evaluation)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Graphics
from tqdm import tqdm

def get_json_data(url):
    '''
    Scrape data (which must be json format) from given url
    Input: url to financialmodelingprep API
    Output: json file
    '''
    response = urlopen(url)
    dat = response.read().decode('utf-8')
    return json.loads(dat)

def get_price_var(symbol):
    '''
    Get historical price data for a given symbol leveraging the power of pandas_datareader and Yahoo.
    Compute the difference between first and last available time-steps in terms of Adjusted Close price..
    Input: ticker symbol
    Output: price variation
    '''
    # read data
    prices = data.DataReader(symbol, 'yahoo', '2019-01-01', '2019-12-31')['Adj Close']

    # get all timestamps for specific lookups
    today = prices.index[-1]
    start = prices.index[0]

    # calculate percentage price variation
    price_var = ((prices[today] - prices[start]) / prices[start]) * 100
    return price_var

def find_in_json(obj, key):
    '''
    Scan the json file to find the value of the required key.
    Input: json file
           required key
    Output: value corresponding to the required key
    '''
    # Initialize output as empty
    arr = []

    def extract(obj, arr, key):
        '''
        Recursively search for values of key in json file.
        '''
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

#getting company bio and profile and industry
url = 'https://financialmodelingprep.com/api/v3/company/stock/list?apikey=463ee25175316666413b11ee5de9ffa5'
ticks_json = get_json_data(url)
available_tickers = find_in_json(ticks_json, 'symbol')
print (available_tickers)
tickers_sector = []
for tick in tqdm(available_tickers):
    url = 'https://financialmodelingprep.com/api/v3/company/profile/' + tick + '?apikey=463ee25175316666413b11ee5de9ffa5' # get sector from here
    a = get_json_data(url)
    tickers_sector.append(find_in_json(a, 'sector'))

S = pd.DataFrame(tickers_sector, index=available_tickers, columns=['Sector'])

# Get list of tickers from TECHNOLOGY sector
S.to_csv('ticker_sector.csv', index = None, header=True)


#pick your own industry here
tickers_tech = S[S['Sector'] == 'Technology'].index.values.tolist()



pvar_list, tickers_found = [], []
num_tickers_desired = 1000
count = 0
tot = 0
TICKERS = tickers_tech

for ticker in TICKERS:
    tot += 1
    try:
        pvar = get_price_var(ticker)
        pvar_list.append(pvar)
        tickers_found.append(ticker)
        count += 1
    except:
        pass

    stdout.write(f'\rScanned {tot} tickers. Found {count}/{len(TICKERS)} usable tickers (max tickets = {num_tickers_desired}).')
    stdout.flush()

    if count == num_tickers_desired: # if there are more than 1000 tickers in sectors, stop
        break

# Store everything in a dataframe
D = pd.DataFrame(pvar_list, index=tickers_found, columns=['2019 PRICE VAR [%]'])


indicators = []
filename = 'indicators.txt'
with open(filename, 'r') as f:
    for line in f:
        indicators.append(line.strip('\n'))
        
missing_tickers, missing_index = [], []
d = np.zeros((len(tickers_found), len(indicators)))

#getting fundamental information
for t, _ in enumerate(tqdm(tickers_found)):
    # Scrape indicators from financialmodelingprep API
    url0 = 'https://financialmodelingprep.com/api/v3/financials/income-statement/' + tickers_found[t]+ '?apikey=463ee25175316666413b11ee5de9ffa5'
    url1 = 'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/' + tickers_found[t]+ '?apikey=463ee25175316666413b11ee5de9ffa5'
    url2 = 'https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/' + tickers_found[t]+ '?apikey=463ee25175316666413b11ee5de9ffa5'
    url3 = 'https://financialmodelingprep.com/api/v3/financial-ratios/' + tickers_found[t]+ '?apikey=463ee25175316666413b11ee5de9ffa5'
    url4 = 'https://financialmodelingprep.com/api/v3/company-key-metrics/' + tickers_found[t]+ '?apikey=463ee25175316666413b11ee5de9ffa5'
    url5 = 'https://financialmodelingprep.com/api/v3/financial-statement-growth/' + tickers_found[t]+ '?apikey=463ee25175316666413b11ee5de9ffa5'
    a0 = get_json_data(url0)
    a1 = get_json_data(url1)
    a2 = get_json_data(url2)
    a3 = get_json_data(url3)
    a4 = get_json_data(url4)
    a5 = get_json_data(url5)
    
    # Combine all json files in a list, so that it can be scanned quickly
    A = [a0, a1, a2 , a3, a4, a5]
    all_dates = find_in_json(A, 'date')

    check = [s for s in all_dates if '2018' in s] # find all 2018 entries in dates
    if len(check) > 0:
        date_index = all_dates.index(check[0]) # get most recent 2018 entries, if more are present

        for i, _ in enumerate(indicators):
            ind_list = find_in_json(A, indicators[i])
            try:
                d[t][i] = ind_list[date_index]
            except:
                d[t][i] = np.nan # in case there is no value inserted for the given indicator

    else:
        missing_tickers.append(tickers_found[t])
        missing_index.append(t)

actual_tickers = [x for x in tickers_found if x not in missing_tickers]
d = np.delete(d, missing_index, 0)
DATA = pd.DataFrame(d, index=actual_tickers, columns=indicators) # raw dataset

DATA = DATA.loc[:, DATA.isin([0]).sum() <= 20]

# Remove columns that have more than 15 nan-values
DATA = DATA.loc[:, DATA.isna().sum() <= 15]

# Fill remaining nan-values with column mean value
DATA = DATA.apply(lambda x: x.fillna(x.mean()))

# Get price variation data only for tickers to be used
D2 = D.loc[DATA.index.values, :]
DATA.to_csv('data_tech_fundamental.csv', index = None, header=True)
#print (DATA)


