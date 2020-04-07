import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('.')))

import pandas as pd

from filter_methods.methods.infogain import information_gain
from filter_methods.methods.cfs import cfs
from filter_methods.methods.relieff import relieff
from filter_methods.methods.mrmr import mrmr
from filter_methods.methods.chi_squared import chi_squared
from filter_methods.methods.fisher_lda import fisher_lda
from dl_methods.methods.TSFS_m import TSFS

from weka.core.converters import Loader, Instances


alg_names = ['CFS', 'CHI2', 'IG', 'LDA', 'MCFS', 'MRMR', 'RELIEFF']


def filter_fs(alg_name, X, y, feat_names, num_features=10, weka_data=None):
    if alg_name not in alg_names:
        print(alg_name, 'is not a supported algorithm. Skipped.')
        return None

    if alg_name in ['CFS', 'IG', 'RELIEFF']:
        # load data for WEKA
        loader = Loader("weka.core.converters.CSVLoader")
        data = loader.load_file(weka_data)
        data.class_is_last()
        filter_data = Instances.copy_instances(data)

        if alg_name == 'CFS':
            results = cfs(filter_data, feat_names)
        if alg_name == 'IG':
            results = information_gain(filter_data, feat_names)
        if alg_name == 'RELIEFF':
            results = relieff(filter_data, feat_names)

        return {alg_name: results}

    if alg_name == 'CHI2':
        results = chi_squared(X, y, feat_names, num_features)

        return {alg_name: results}

    if alg_name == 'LDA':
        results = TSFS(X, y, feat_names, 'MCFS')

        return {alg_name: results}

    if alg_name == 'MCFS':
        results = fisher_lda(X, y, feat_names)

        return {alg_name: results}

    if alg_name == 'MRMR':
        results = mrmr(X, y, feat_names, num_features)

        return {alg_name: results}

