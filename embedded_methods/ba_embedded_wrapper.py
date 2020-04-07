import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('.')))

from embedded_methods.methods.elastic_net import elastic_net
from embedded_methods.methods.hsic_lasso import hsic
from embedded_methods.methods.lasso import lasso
from embedded_methods.methods.stability_selection import stability_selection

alg_names = ['EN', 'HSIC', 'LASSO', 'SS']


def embedded_fs(alg_name, X, y, feat_names, num_features=10, hsic_data=None):
    if alg_name not in alg_names:
        print(alg_name, 'is not a supported algorithm. Skipped.')
        return None

    if alg_name == 'EN':
        results = elastic_net(X, y, feat_names)

        return {alg_name: results}

    if alg_name == 'HSIC':
        results = hsic(num_features, hsic_data)

        return {alg_name: results}

    if alg_name == 'LASSO':
        results = lasso(X, y, feat_names)

        return {alg_name: results}

    if alg_name == 'SS':
        results = stability_selection(X, y, feat_names)

        return {alg_name: results}


