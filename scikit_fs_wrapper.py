import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from wrapper_methods.methods.jackstraw_m import jackstraw
from wrapper_methods.methods.boruta import boruta_fs
from wrapper_methods.methods.svm_rfe import svm_rfe

from pandas.api.types import is_numeric_dtype
from weka.core.converters import Loader, Instances

def mmr_wrapper(X, y, feat_names, num_fea):
    indexes,_,_ = MRMR.mrmr(X, np.ravel(y), n_selected_features=num_fea)
    results = [feat_names[idx] for idx in indexes]
    return results


def non_dl_wrapper(alg_name, X, y, feat_names):
    features = pd.DataFrame(X, columns=feat_names)
    target = pd.DataFrame(y, columns=['class'])
    # target['class'] = pd.factorize(target['class'])[0] + 1

    if alg_name == 'boruta':
        return {alg_name : boruta_fs( X, y, feat_names)}


    print('Error Embedded wrapper: invalid name: ', alg_name)

    return {alg_name : []}