import pandas as pd
import numpy as np
import pickle
import optuna
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "svg" # Toggle for GitHub rendering

from mlfinlab.online_portfolio_selection import *

data = pd.read_csv('sp_500.csv', parse_dates=True, index_col='date')
data = data.dropna(axis=1)
#data = data.drop(data.columns[100:400],axis=1)
equity_corn = optuna.load_study(study_name='corn', storage='sqlite:///stored/equity.db')
equity_cornu = optuna.load_study(study_name='cornu', storage='sqlite:///stored/equity.db')
equity_scorn = optuna.load_study(study_name='scorn', storage='sqlite:///stored/equity.db')
equity_fcorn = optuna.load_study(study_name='fcorn', storage='sqlite:///stored/equity.db')

with open('pickled/equity_cornk.pickle', 'rb') as fout:
    equity_cornk = pickle.load(fout)
equity_cornk = pd.DataFrame(equity_cornk, columns=['Window','Rho','K','Returns']).sort_values('Returns', ascending=False)

equity_cornk_ = CORNK(window=2, rho=5, k=2)
equity_cornk_.allocate(data, verbose=True)

with open('pickled/equity_scornk.pickle', 'rb') as fout:
    equity_scornk = pickle.load(fout)
equity_scornk = pd.DataFrame(equity_scornk, columns=['Window','Rho','K','Returns']).sort_values('Returns', ascending=False)

equity_scornk_ = SCORNK(window=2, rho=1, k=2)
equity_scornk_.allocate(data, verbose=True)

equity_fcornk = FCORNK(window=2, rho=1, lambd=1, k=2)
equity_fcornk.allocate(data, verbose=True)

equity_bah = BAH()
equity_bah.allocate(data)

# Constant Rebalanced Portfolio.
equity_crp = CRP()
equity_crp.allocate(data)

equity_corn_ = CORN(window=equity_corn.best_params['window'], rho=equity_corn.best_params['rho'])

# CORN-U.
equity_cornu_ = CORNU(window=equity_cornu.best_params['window'], rho=equity_cornu.best_params['rho'])

# SCORN.
equity_scorn_ = SCORN(window=equity_scorn.best_params['window'], rho=equity_scorn.best_params['rho'])

# FCORN.
equity_fcorn_ = FCORN(window=equity_fcorn.best_params['window'], rho=equity_fcorn.best_params['rho'], lambd=equity_fcorn.best_params['lambd'])

equity_corn_.allocate(data)
equity_cornu_.allocate(data)
equity_scorn_.allocate(data)
equity_fcorn_.allocate(data)

fig = go.Figure()
idx = equity_bah.portfolio_return.index
fig.add_trace(go.Scatter(x=idx, y=equity_bah.portfolio_return['Returns'], name="Buy and Hold"))
fig.add_trace(go.Scatter(x=idx, y=equity_crp.portfolio_return['Returns'], name="CRP"))
fig.add_trace(go.Scatter(x=idx, y=equity_corn_.portfolio_return['Returns'], name="CORN"))
fig.add_trace(go.Scatter(x=idx, y=equity_cornu_.portfolio_return['Returns'], name="CORN-U"))
fig.add_trace(go.Scatter(x=idx, y=equity_cornk_.portfolio_return['Returns'], name="CORN-K"))
fig.add_trace(go.Scatter(x=idx, y=equity_scorn_.portfolio_return['Returns'], name="SCORN"))
fig.add_trace(go.Scatter(x=idx, y=equity_scornk_.portfolio_return['Returns'], name="SCORN-K"))
fig.add_trace(go.Scatter(x=idx, y=equity_fcorn_.portfolio_return['Returns'], name="FCORN"))
#fig.add_trace(go.Scatter(x=idx, y=equity_fcornk.portfolio_return['Returns'], name="FCORN-K"))

fig.update_layout(title='Pattern Matching Strategies on US Equity', xaxis_title='Date', yaxis_title='Relative Returns')
fig.show()
