#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:15:28 2020

@author: zhangwenyong
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from scipy import stats
import sys
sys.setrecursionlimit(1000000)



class Trends():
    def __init__(self, df, timescale = 1024, cap=False):
        '''
        Arguments:
        df: dataframe including close price
        timescale: 2^k \in {1,2,3,4,5,6,7,8,9,10}
        cap:whether using clip
        '''
        self.df = df    
        self.df_close = df['Adj Close']    
        self.log_return = np.log(self.df_close/self.df_close.shift(1))
        self.mu = self.log_return.mean()
        self.sigma = self.log_return.std()
        #df_close = df['Adj Close']
        #self.df['log return'] = np.log(df_close/df_close.shift(1))
        #r = df['log return'].values
        #mu = df['log return'].mean()
        #sigma = df['log return'].std()
        self.Return = self.log_return/self.sigma
        self.df['ExcessReturn'] = (self.Return - self.mu/self.sigma).fillna(0)
        #Return = r/sigma
        #df['ExcessReturn'] = Return - mu/sigma
        #df['ExcessReturn'] = df['ExcessReturn'].fillna(0)
        self.ExcessReturn = df['ExcessReturn'].values
        self.T = timescale
        self.cap = cap
        
        def N(T):
            return ((1- np.exp(-4/T))**2)/np.sqrt(1-np.exp(-8/T))
        self.N = N
        def M(T):
            return np.sqrt(1 - np.exp(-4/T))
        self.M = M    
        
        self.weights = self.weight(self.T)
        self.weights_sai = self.weight_sai(self.T)
        
        Total_Phi = [0]
        for t in range(1,self.df['ExcessReturn'].values.shape[0]+1):
            Total_Phi.append(self.phi(t))
        
        self.Total_Phi = np.array(Total_Phi)
        
        Total_Sai = [0]
        for t in range(1,self.df['ExcessReturn'].values.shape[0]+1):
            Total_Sai.append(self.sai(t))
        
        self.Total_Sai = np.array(Total_Sai)
        
    def weight(self, T): #calculate weight for whole time horizon
        weights = []
        for n in range(self.df['ExcessReturn'].values.shape[0]):
            w = self.N(T)*(n+1)*np.exp(-2*n/T)
            weights.append(w)
        weights.reverse()
        weights = np.array(weights)
        return weights
        
        
    def weight_sai(self, T): #calculate exp{-2n/T} for whole time horizon
        weights = []
        for n in range(self.df['ExcessReturn'].values.shape[0]):
            w = np.exp(-2*n/T)
            weights.append(w)
        weights.reverse()
        weights = np.array(weights)
        return weights
    
        
        
        
    def phi(self, t): #each phi_t for each time step,phi_t is analytical
        if t == 0: 
            c=0
        else:
          return_series = self.ExcessReturn[:t]
          weights_we_want = self.weights[-t:]
          c = np.sum(return_series*weights_we_want)
        return c
            
    def sai(self, t): #each sai_t for each time step
        if t == 0: 
            c=0
        else:
          return_series = self.ExcessReturn[:t]
          weights_we_want = self.weights_sai[-t:]
          c = np.sum(return_series*weights_we_want)
        return c
        
        
        
        
        Total_Phi = [0]
        for t in range(1,self.df['ExcessReturn'].values.shape[0]+1):
            Total_Phi.append(self.phi(t))
        
        self.Total_Phi = np.array(Total_Phi)
        
        Total_Sai = [0]
        for t in range(1,self.df['ExcessReturn'].values.shape[0]+1):
            Total_Sai.append(self.sai(t))
        
        self.Total_Sai = np.array(Total_Sai)
        
        
    def Adjust_phi(self, t):
        if t == 0:
            return 0
        elif t == 1:
            return self.Total_Phi[1]
        else:
            if self.cap == True:
              sai = min(2.5,max(-2.5, self.Total_Sai[t])) #do the clip
            else:
              sai = self.Total_Sai[t]
        return sai*self.N(self.T)/self.M(self.T) + self.Adjust_phi(t-1)*np.exp(-2/self.T)
        
    def get_phi(self):
        Phi = []
        for t in range(1,self.df['ExcessReturn'].values.shape[0]+1):
            Phi.append(self.phi(t))
        Phi = np.array(Phi)    
        
        self.df['Phi'] = Phi
        self.df['Phi'].plot()
    
    
    def get_trends(self):
        Adjust_Phi = []
        for t in range(1,self.df['ExcessReturn'].values.shape[0]+1):
            Adjust_Phi.append(self.Adjust_phi(t))
        Adjust_Phi = np.array(Adjust_Phi)    
        
        self.df['Adjust trend'] = Adjust_Phi
        self.df['Adjust trend'].plot()
        

if __name__ == '__main__':
    tick_data = pd.read_csv('AMD_dollar_all.csv', parse_dates=['date_time']).set_index('date_time')
    
    timeframe = '1D'
    
    ohlc_data = tick_data['close'].resample(timeframe).ohlc()
    
    ohlc_data['volume'] = tick_data['cum_vol'].resample(timeframe).sum()
    
    print (ohlc_data)
    
    trends = Trends(ohlc_data, 1024)
    trends.get_trends()
    

    



