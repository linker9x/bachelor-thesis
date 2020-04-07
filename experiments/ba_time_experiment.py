import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('.')))

import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import StratifiedKFold
from filter_methods.ba_filter_wrapper import filter_fs
from embedded_methods.ba_embedded_wrapper import embedded_fs
from wrapper_methods.ba_wrapper_wrapper import wrapper_fs
from dl_methods.ba_dl_wrapper import deepl_fs
from evaluator.timer import timer

import weka.core.jvm as jvm

split = 2
filepath = '../data/'
dir_out = './results/time'
param_methods = ['MRMR', 'CHI2', 'CAE', 'HSIC']
filter_methods = ['CFS', 'CHI2', 'IG', 'LDA', 'MCFS', 'MRMR', 'RELIEFF']
embedded_methods = ['EN', 'HSIC', 'LASSO', 'SS']
wrapper_methods = ['BORUTA', 'JACKSTRAW', 'SVM-RFE']
dl_methods = ['CAE', 'MLP', 'SCA', 'DBN', 'TSFS']


def run_multi_time_experiment(alg_names, ds_names, num_feats=250):
    try:
        jvm.start()

        # datasets to use
        ds_names = ds_names

        for alg_name in alg_names:
            times = pd.DataFrame(columns=['alg_name', 'ds_name', 'time_in_sec'])
            for ds_name in ds_names:
                df = pd.read_csv(filepath + ds_name + '_clean.csv')

                y = df.pop('class').values
                feature_names = np.array(df.columns.values)
                X = df.values

                # cross validation training and test splits
                cv = StratifiedKFold(n_splits=split, random_state=42, shuffle=False)

                train_index, test_index = cv.split(X, y)

                print('Alg: %s, DS: %s\n' % (alg_name, ds_name))

                X_train, X_test, y_train, y_test = X[train_index[0]], X[test_index[0]], y[train_index[0]], y[test_index[0]]

                if alg_name == 'HSIC':
                    train_filepath = save_train_csv(X_train, y_train, feature_names, 0, ds_name, HSIC=True)
                elif alg_name in ['CFS', 'IG', 'RELIEFF']:
                    train_filepath = save_train_csv(X_train, y_train, feature_names, 0, ds_name)
                else:
                    train_filepath = None

                # run fs alg for each fold for non-subset methods
                if alg_name not in param_methods:
                    print('Normal')
                    alg_time = timer()
                    if alg_name in filter_methods:
                        new_vals = filter_fs(alg_name, X_train, y_train, feature_names, weka_data=train_filepath)
                        alg_time = timer(alg_time)
                    elif alg_name in embedded_methods:
                        new_vals = embedded_fs(alg_name, X_train, y_train, feature_names)
                        alg_time = timer(alg_time)
                    elif alg_name in wrapper_methods:
                        new_vals = wrapper_fs(alg_name, X_train, y_train, feature_names)
                        alg_time = timer(alg_time)
                    elif alg_name in dl_methods:
                        new_vals = deepl_fs(alg_name, X_train, y_train, feature_names)
                        alg_time = timer(alg_time)
                    else:
                        raise ValueError('%s is not a supported algorithm.' % alg_name)


                if alg_name in param_methods:
                    print('Param')
                    alg_time = timer()
                    if alg_name in filter_methods:
                        new_vals = filter_fs(alg_name, X_train, y_train, feature_names, num_feats, weka_data=train_filepath)
                        alg_time = timer(alg_time)
                    elif alg_name in embedded_methods:
                        new_vals = embedded_fs(alg_name, X_train, y_train, feature_names, num_feats, hsic_data=train_filepath)
                        alg_time = timer(alg_time)
                    elif alg_name in wrapper_methods:
                        new_vals = wrapper_fs(alg_name, X_train, y_train, feature_names)
                        alg_time = timer(alg_time)
                    elif alg_name in dl_methods:
                        new_vals = deepl_fs(alg_name, X_train, y_train, feature_names, num_features=num_feats)
                        alg_time = timer(alg_time)
                    else:
                        raise ValueError('%s is not a supported algorithm.' % alg_name)


                print('Time: %s\n' % (alg_time))

                entry = pd.DataFrame({'alg_name': [alg_name],
                                      'ds_name': [ds_name],
                                      'time': [alg_time]
                                      })
                times = times.append(entry, ignore_index=True)

            times.to_csv('./results/' + alg_name + 'time.csv', index=None, header=True)

    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()



def save_train_csv(X_train, y_train, feature_names, fold_idx, ds_name, HSIC=False):
    dir_csv =  dir_out + ds_name + '/'
    if not os.path.exists(dir_csv):
        os.mkdir(dir_csv)

    df_x = pd.DataFrame(X_train, columns=feature_names)
    df_y = pd.DataFrame(y_train, columns=['class'])
    if not HSIC:
        df_y['class'] = 'class_' + df_y['class'].astype(str)

    fold_df = pd.concat([df_x, df_y], axis=1)
    fold_train_path = dir_csv + str(fold_idx) + '.csv'
    fold_df.to_csv(fold_train_path, index=False)
    return fold_train_path

