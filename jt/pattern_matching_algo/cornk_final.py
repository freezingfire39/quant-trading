import pandas as pd
import numpy as np
import sys
import pickle
import optuna
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "svg" # Toggle for GitHub rendering

from mlfinlab.online_portfolio_selection import *

# input_path = 'sp_500.csv'
input_path = sys.argv[1]
output_path = sys.argv[2]

data = pd.read_csv(input_path, parse_dates=True, index_col='date')
data = data.drop(data.columns[50:400],axis=1)
data.dropna(inplace=True)

with open('pickled/equity_cornk.pickle', 'rb') as fout:
    equity_cornk = pickle.load(fout)
equity_cornk = pd.DataFrame(equity_cornk, columns=['Window','Rho','K','Returns']).sort_values('Returns', ascending=False)

equity_cornk = optuna.load_study(study_name='cornk', storage='sqlite:///TestDB.db')
equity_cornk_ = CORNK(window=equity_cornk.best_params['window'], rho=equity_cornk.best_params['rho'],k=equity_cornk.best_params['k'])
equity_cornk_.allocate(data)
positions = equity_cornk_.all_weights
positions.to_csv (output_path, header=True)
