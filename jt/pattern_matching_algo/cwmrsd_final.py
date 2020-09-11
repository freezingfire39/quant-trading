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

equity_cwmrsd_ = optuna.load_study(study_name='cwmr', storage='sqlite:///TestDB.db')
equity_cwmrsd_ = CWMR(confidence=equity_cwmrsd_.best_params['confidence'], epsilon=equity_cwmrsd_.best_params['epsilon'], method='sd')
equity_cwmrsd_.allocate(data)

positions = equity_cwmrsd_.all_weights
# positions.to_csv ('weights_cwmr.csv', header=True)
positions.to_csv (output_path, header=True)
