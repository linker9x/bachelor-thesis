import warnings
warnings.filterwarnings('ignore')
    
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('.')))

import pandas as pd
pd.set_option('display.max_rows', 1000)

from experiments.ba_time_experiment import run_multi_time_experiment


alg_names = ['MCFS']

ds_names = ['CLL_SUB_111', 'CNS', 'colon', 'GLI_85', 'GLIOMA', 'Leukemia', 'lung',
            'Lymphoma', 'MLL', 'ovarian', 'Prostate_GE', 'SRBCT', 'TOX_171']

# alg_names = ['EN', 'IG']
#
# ds_names = ['colon', 'GLIOMA']

run_multi_time_experiment(alg_names, ds_names)
