#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:26:35 2020

@author: zhangwenyong
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nfft import nfft
from sklearn.linear_model import LinearRegression


def LSp(w, f_hat): #Lomb-Scargle periodogram
    N = len(f_hat)
    t = np.arange(0, 1, 1/N)
    sigma = f_hat.std()
    mu = f_hat.mean()
    P = []
    for i in range(w.shape[0]):
      tau = np.arctan((np.sum(np.sin(2*w[i]*t))/np.sum(np.cos(2*w[i]*t))))/w[i]/2
     
      N1 = (((f_hat-mu)*np.cos(w[i]*(t-tau))).sum())**2
      D1 = 2*sigma**2*((np.cos(w[i]*(t-tau)))**2).sum()
      N2 = (((f_hat-mu)*np.sin(w[i]*(t-tau))).sum())**2
      D2 = 2*sigma**2*((np.sin(w[i]*(t-tau)))**2).sum()
    
      p = N1/D1 + N2/D2
      P.append(p)
    P = np.array(P)
    return P
       


class Transform():
    def __init__(self, df, T=254):
        '''
        Our aim is to get a feature from Non-uniform FFT on daily return data,
        Then we extract features and target and then combine them as the data 
        that can be used in machine learning.
        '''
        self.df = df
        self.T = T #T must be even!!!
        self.df_close = df['close']    
        self.log_return = np.log(self.df_close/self.df_close.shift(1))
        self.mu = self.log_return.mean()
        self.sigma = self.log_return.std()
        self.Return = self.log_return/self.sigma
        self.df['ExcessReturn'] = (self.Return - self.mu/self.sigma).fillna(0)
        self.ExcessReturn = df['ExcessReturn'].values
    
    
    def get_feats(self, normalized = False, method = 'NUFFT'): #Lomb-Scargle
        feat = []
        dfreturn = self.df['ExcessReturn']
        for t in range(dfreturn.shape[0]):
            if t <= self.T:
                feat.append(0)
            else:
                x = np.arange(0.01, 1, 1/self.T)
                f_hat = dfreturn[t-self.T:t]
                if method == 'NUFFT':
                  f = abs(nfft(x, f_hat))
                  
                else:
                  f = LSp(x, f_hat)
                feat.append(max(f))
        feat = np.array(feat)
        if normalized == False:
         self.df['freq_feat'] = feat
        else:
         self.df['freq_feat'] = (feat - feat.mean())/feat.std()
        
        
    def plot_freq(self, T): #T mustbe even
        x = np.arange(0, 1, 1/T)
        f_hat = self.df['ExcessReturn'][:T]
        f = abs(nfft(x, f_hat))
        plt.figure()
        plt.plot(x, f)
        plt.show()
        g = LSp(x, f_hat)
        plt.figure()
        plt.plot(x, g)
        plt.show()
            
         
    def get_MLdata(self): #run this function after get features
        y = self.df['ExcessReturn'][self.T:].values
        X = self.df['freq_feat'][self.T:].values
        w = np.arange(len(y))
        w = w/w.sum()
        y = y.reshape(-1,1)
        X = X.reshape(-1,1)
        w = w.reshape(-1,1)
        return y, X, w
        
    
if __name__ == '__main__':
    tick_data = pd.read_csv('AMD_dollar_all.csv', parse_dates=['date_time']).set_index('date_time')
    
    timeframe = '1D'
    
    ohlc_data = tick_data['close'].resample(timeframe).ohlc()
    
    ohlc_data['volume'] = tick_data['cum_vol'].resample(timeframe).sum()
    
    print (ohlc_data)
    
    tfm = Transform(ohlc_data)
    tfm.plot_freq(254)
    tfm.get_feats(normalized = True, method = 'NUFFT')
    y, X, _ = tfm.get_MLdata()
    plt.scatter(X, y, alpha=0.6)
    
    reg = LinearRegression().fit(X, y)
    
    
    
    
    
    
    
    
    
'''    
tick_data = pd.read_csv('amd_dollar_all.csv', parse_dates=['date_time']).set_index('date_time')

timeframe = '1d'

ohlc_data = tick_data['close'].resample(timeframe).ohlc()

ohlc_data['volume'] = tick_data['cum_vol'].resample(timeframe).sum()

print (ohlc_data)


df = ohlc_data
df_close = df['close']   
log_return = np.log(df_close/df_close.shift(1))
mu = log_return.mean()
sigma = log_return.std()
Return = log_return/sigma
df['ExcessReturn'] = (Return - mu/sigma).fillna(0)
dfreturn = df['ExcessReturn'].values
T = 254
f_hat = dfreturn[:T]
x = np.arange(0,1, 1/T)
f = nfft(x, f_hat)
f_ = abs(f)
plt.plot(x, f_)
xstar = x[np.argmax(f_)]


feat = []
for t in range(dfreturn.shape[0]):
    L = len(dfreturn[:t])
    T_ = min(L, T)
    if t <= T:
      feat.append(0)
    else:
      x = np.arange(0, 1, 1/T_)
      f_hat = dfreturn[t-T_:t]
      f = abs(nfft(x, f_hat))
      feat.append(max(f))
feat = np.array(feat)
     
df['freq_feat'] = feat
    
'''













'''

if __name__ == '__main__':
    tick_data = pd.read_csv('AMD_dollar_all.csv', parse_dates=['date_time']).set_index('date_time')
    
    timeframe = '1D'
    
    ohlc_data = tick_data['close'].resample(timeframe).ohlc()
    
    ohlc_data['volume'] = tick_data['cum_vol'].resample(timeframe).sum()
    
    print (ohlc_data)
'''
    
    
    
    
    
    
    
    
    
    
    