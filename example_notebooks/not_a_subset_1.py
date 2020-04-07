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

from experiments.ba_subset_experiment import run_multi_subset_experiment


# returns unranked subset
alg_names = ['HSIC']

#returns ranked subset
# alg_names = ['EN', 'LASSO', 'SS']

# returns param amount of feats - return 1 to 275
# alg_names = ['HSIC']
# , 'CHI2', 'CAE', 'HSIC']

# returns all features ranked - return 1 to 275
# alg_names = ['IG', 'LDA', 'MCFS', 'RELIEFF', 'BORUTA', 'SVM-RFE', 'MLP', 'SCA', 'DBN', 'TSFS']

ds_names = ['GLI_85', 'Lymphoma', 'lung', 'GLIOMA', 'MLL', 'ovarian', 'TOX_171', 'CLL_SUB_111', 'colon', 'SRBCT', 'Prostate_GE', 'CNS', 'Leukemia']

run_multi_subset_experiment(alg_names, ds_names)