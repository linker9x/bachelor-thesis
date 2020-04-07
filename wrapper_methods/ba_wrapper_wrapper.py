def wrapper_fs(alg_name, X_train, y_train, feature_names):
    return None
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('.')))

from wrapper_methods.methods.jackstraw_m import jackstraw
from wrapper_methods.methods.boruta import boruta_fs
from wrapper_methods.methods.svm_rfe import svm_rfe


alg_names = ['BORUTA', 'JACKSTRAW', 'SVM-RFE']


def wrapper_fs(alg_name, X, y, feat_names):
    if alg_name not in alg_names:
        print(alg_name, 'is not a supported algorithm. Skipped.')
        return None

    if alg_name == 'BORUTA':
        results = boruta_fs(X, y, feat_names)

        return {alg_name: results}

    if alg_name == 'JACKSTRAW':
        results = jackstraw(X, feat_names)

        return {alg_name: results}

    if alg_name == 'SVM-RFE':
        results = svm_rfe(X, y, feat_names)

        return {alg_name: results}


