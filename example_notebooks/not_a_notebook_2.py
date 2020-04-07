import warnings
warnings.filterwarnings('ignore')
    
import os
import traceback
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('.')))

import pandas as pd
pd.set_option('display.max_rows', 1000)
import numpy as np 
import matplotlib.pyplot as plt

from experiments.ba_experiment import run_multi_experiment

num_feats = [5, 10, 15, 20, 25, 35, 45, 55, 65, 75, 125, 150, 175, 200, 250]

# returns unranked subset - (think)
# alg_names = ['JACKSTRAW', 'CFS']

#returns ranked subset - (think)
# alg_names = ['EN', 'LASSO', 'SS']

# returns param amount of feats - return 1 to 250
# alg_names = ['MRMR', 'CHI2', 'CAE', 'HSIC']
alg_names = ['MRMR']

# returns all features ranked - return 1 to 250
# alg_names = ['DBN', 'TSFS']

ds_names = ['SRBCT', 'ovarian']

run_multi_experiment(alg_names, ds_names, num_feats)