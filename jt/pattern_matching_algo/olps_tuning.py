  
"""
Optuna tuning script for OLPS strategies.
1. Set number of iterations.
2. Set name of the study.
3. Set the data.
4. Set optuna study.
5. Create objective function. (More available from olps_create_obj.py).
6. Run study.
"""
import pandas as pd
import numpy as np
import optuna
from mlfinlab.online_portfolio_selection import *
import joblib

# Number of iterations for parameters.
number = 200

# Study name.
s_name = 'cwmr'

# Set data that you want to use.
data = pd.read_csv('sp_500.csv', parse_dates=True, index_col='date')
data = data.drop(data.columns[50:400],axis=1)
#data = data[~(data.index> '2017-01-01')]

# Load optuna study.
study = optuna.load_study(study_name=s_name,storage='sqlite:///TestDB.db')

# Objective function for CORN. More can be found in olps_create_obj.py.
def obj(trial):
    # Window integer range from 1 to 30.
    confidence = trial.suggest_uniform('confidence', 0, 1)

    # Rho uniform range from -1 to 1.
    epsilon = trial.suggest_uniform('epsilon', 0, 1)

    # Create CORN model with given window and rho.
    model = CWMR(confidence, epsilon, method='sd')
    
    #window = trial.suggest_int('window', 1, 30)

    # Rho uniform range from -1 to 1.
    #rho = trial.suggest_int('rho', 1, 10)  ##if i just set it at 2 it will run
    #k = trial.suggest_int('k', 1, 5)

    # Create CORN model with given window and rho.
    #model = CORNK(window, rho,k)

    # Allocate to model with verbose=True to follow progress.
    model.allocate(data, verbose=True)

    # Record intermdiate steps for future analysis.
    data_len, period = data.shape[0], 10
    for i in range(0, period):
        time = data_len * i // period
        trial.report(model.portfolio_return.iloc[time][0], step=time)

    return model.portfolio_return.iloc[-1][0]

# Optimize for given study.
study.optimize(obj, n_trials=number)
joblib.dump(study, 'cwmr.pkl')
study = joblib.load('cwmr.pkl')
print('Best trial until now:')
print(' Value: ', study.best_trial.value)
print(' Params: ')
for key, value in study.best_trial.params.items():
    print(f'    {key}: {value}')

