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
from evaluator.eval import ba_clf_evaluator

import weka.core.jvm as jvm

split = 5
filepath = '../data/'
dir_out = './results/'
param_methods = ['MRMR', 'CHI2', 'CAE', 'HSIC']
filter_methods = ['CFS', 'CHI2', 'IG', 'LDA', 'MCFS', 'MRMR', 'RELIEFF']
embedded_methods = ['EN', 'HSIC', 'LASSO', 'SS']
wrapper_methods = ['BORUTA', 'JACKSTRAW', 'SVM-RFE']
dl_methods = ['CAE', 'MLP', 'SCA', 'DBN', 'TSFS']


def run_experiment(alg_name, ds_names, num_feats):
    # datasets to use
    ds_names = ds_names

    # array containing number of features to run for
    num_feats = num_feats

    for ds_name in ds_names:
        print('%s: %s' % (alg_name, ds_name))
        df = pd.read_csv(filepath + ds_name + '_clean.csv')

        y = df.pop('class').values
        feature_names = np.array(df.columns.values)
        X = df.values

        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)
        # _, y = change_class_labels(y)
        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)

        # cross validation training and test splits
        cv = StratifiedKFold(n_splits=split, random_state=42, shuffle=False)
        fold_idx = 0                    # track split number
        results = dict()                # dictionary for cumulative results
        fold_returned_dict = dict()     # dictionary for fold results
        selected_feats_dict = dict()    # dictionary for selected features

        for train_index, test_index in cv.split(X, y):
            print('Fold No: %d, Alg: %s, DS: %s\n' % (fold_idx, alg_name, ds_name))
            fold_dict = dict()  # dictionary for fold

            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

            if alg_name == 'HSIC':
                train_filepath = save_train_csv(X_train, y_train, feature_names, fold_idx, ds_name, HSIC=True)
            elif alg_name in ['CFS', 'IG', 'RELIEFF']:
                train_filepath = save_train_csv(X_train, y_train, feature_names, fold_idx, ds_name)
            else:
                train_filepath = None

            # run fs alg for each fold for non-subset methods
            if alg_name not in param_methods:
                print('Normal')
                if alg_name in filter_methods:
                    new_vals = filter_fs(alg_name, X_train, y_train, feature_names, weka_data=train_filepath)
                elif alg_name in embedded_methods:
                    new_vals = embedded_fs(alg_name, X_train, y_train, feature_names)
                elif alg_name in wrapper_methods:
                    new_vals = wrapper_fs(alg_name, X_train, y_train, feature_names)
                elif alg_name in dl_methods:
                    new_vals = deepl_fs(alg_name, X_train, y_train, feature_names)
                else:
                    raise ValueError('%s is not a supported algorithm.' % alg_name)

            # now we evaluate the returned subset with the classifiers
            alg_dict = dict()   # don't know if we need this yet

            # if ranked, we'll pass top x features
            # if not ranked, i.e. subset or param, run alg for each num of features
            for num_feat in num_feats:
                print('No. Feats:', num_feat, '\n')
                # run alg for each num of features
                if alg_name in param_methods:
                    print('Param')
                    if alg_name in filter_methods:
                        new_vals = filter_fs(alg_name, X_train, y_train, feature_names, num_feat, weka_data=train_filepath)
                    elif alg_name in embedded_methods:
                        new_vals = embedded_fs(alg_name, X_train, y_train, feature_names, num_feat, hsic_data=train_filepath)
                    elif alg_name in wrapper_methods:
                        new_vals = wrapper_fs(alg_name, X_train, y_train, feature_names)
                    elif alg_name in dl_methods:
                        new_vals = deepl_fs(alg_name, X_train, y_train, feature_names, num_features=num_feat)
                    else:
                        raise ValueError('%s is not a supported algorithm.' % alg_name)

                # if num_feat 0, pass all, otherwise pass num_feat number of feats
                selected_feats = new_vals[alg_name] if num_feat == 0 else new_vals[alg_name][:num_feat]

                # save the selected features to the dict on last iteration for ranked algs
                if alg_name not in param_methods:
                    if num_feat == num_feats[-1]:  # last element
                        # alg not in selected_feats_dict? add
                        if alg_name not in selected_feats_dict:
                            selected_feats_dict[alg_name] = dict()
                        # fold not in the dict? add
                        if fold_idx not in selected_feats_dict[alg_name]:
                            # save selected feats in selected_feats_dict under fold
                            selected_feats_dict[alg_name][fold_idx] = selected_feats
                # if this is a method that returns a subset or takes a param, do this every iter
                else:
                    # alg not in selected_feats_dict? add
                    if alg_name not in selected_feats_dict:
                        selected_feats_dict[alg_name] = dict()
                    # fold not in the dict? add
                    if fold_idx not in selected_feats_dict[alg_name]:
                        selected_feats_dict[alg_name][fold_idx] = [] # add empty list
                    # save selected feats under fold index
                    selected_feats_dict[alg_name][fold_idx].append('NO_FEAT_' + str(num_feat))
                    selected_feats_dict[alg_name][fold_idx].extend(selected_feats)

                # set up for fold dict
                # alg not in fold_returned_dict? add
                if alg_name not in fold_returned_dict:
                    fold_returned_dict[alg_name] = dict()
                # num_feat not in fold_returned_dict? add
                if num_feat not in fold_returned_dict[alg_name]:
                    fold_returned_dict[alg_name][num_feat] = list()
                # save selected feats for number of feats to dict
                fold_returned_dict[alg_name][num_feat].append(selected_feats)

                # calculate scores for classifiers
                scores = ba_clf_evaluator(X_train, X_test, y_train, y_test, feature_names, selected_feats)
                # save scores for number of feats to alg_dict
                alg_dict[num_feat] = scores

            fold_dict[alg_name] = alg_dict
            results[fold_idx] = fold_dict
            fold_idx += 1   # next split

        print('SAVING')
        write_results(results, alg_name, dir_out + ds_name + '/', selected_feats_dict)

    return results


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


def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir


def write_results(results, alg_name, dir_out, selected_feats_dict):
    dir_out = standardize_dir(dir_out)

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    writers = dict()

    alg_out_path = dir_out + str(alg_name) + '.csv'  # file for alg scores
    writers[alg_name] = open(alg_out_path, 'w')      # write
    writers[alg_name].write('alg_name,fold_id,num_feat,clf_name,acc,pre,rec,f1,class_pre,class_rec,class_f1\n')
    # for every fold
    for fold_idx in results.keys():
        # get the results for the index
        fold_results = results[fold_idx]
        # get the results for the alg
        alg_fold_res = fold_results[alg_name]
        # for each returned subset
        for num_feat in sorted(alg_fold_res.keys()):
            feat_res = alg_fold_res[num_feat]
            if feat_res == {}:
                continue
            # write data for each entry
            for clf_name in feat_res.keys():
                writers[alg_name].write(str(alg_name) + ',' + str(fold_idx) + ',' + str(num_feat) + ',' + str(clf_name))
                for i_tmp in feat_res[clf_name]:
                    writers[alg_name].write(',' + str(i_tmp))
                writers[alg_name].write('\n')

    # close all the writers
    for alg in writers.keys():
        writers[alg].close()


    selected_writer_path = dir_out + alg_name + '_selected_feat.csv'
    selected_writer = open(selected_writer_path, 'w')
    for fold_idx in range(split):
        selected_writer.write(','.join(selected_feats_dict[alg_name][fold_idx]) + '\n')
    selected_writer.close()

def run_multi_experiment(alg_names, ds_names, num_feats):
    try:
        jvm.start()

        for alg_name in alg_names:
            run_experiment(alg_name, ds_names, num_feats)

    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
