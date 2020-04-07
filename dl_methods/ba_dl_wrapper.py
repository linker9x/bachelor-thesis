import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('.')))

import pandas as pd

from dl_methods.methods.MLP import MLP
from dl_methods.methods.DBN import DBN
from dl_methods.methods.SCA import SCA
from dl_methods.methods.conc_autoenc import CAE
from dl_methods.methods.TSFS_m import TSFS


alg_names = ['CAE', 'MLP', 'SCA', 'DBN', 'TSFS']


def deepl_fs(alg_name, X, y, feat_names, num_features=10, method_name='my_tsne'):
    if alg_name not in alg_names:
        print(alg_name, 'is not a supported algorithm. Skipped.')
        return None

    if alg_name == 'CAE':
        results = CAE(X, y, feat_names, num_features)

        return {alg_name: results}

    if alg_name == 'MLP':
        results = MLP(X, y, feat_names)

        return {alg_name: results}

    if alg_name == 'SCA':
        results = SCA(X, y, feat_names)

        return {alg_name: results}

    if alg_name == 'DBN':
        results = DBN(X, y, feat_names)

        return {alg_name: results}

    if alg_name == 'TSFS':
        results = TSFS(X, y, feat_names, method_name)

        return {alg_name: results}

