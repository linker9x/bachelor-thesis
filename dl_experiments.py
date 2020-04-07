import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import zscore


import numpy as np

from dl_methods.dl_wrapper import *
from evaluator.eval import all_clf_evaluator
from dl_methods.methods.dbn_mlp_sca.deep_learning.classification import change_class_labels
import os

def cross_validate(csv_path, dir_out, random_state=42, normalize = False):
    df = pd.read_csv(csv_path)
    df['class'] = pd.factorize(df['class'])[0] + 1
    y = df.pop('class').values
    if normalize:
        df = df.apply(zscore)
    feature_names = np.array(df.columns.values)
    X = df.values
    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)
    _, y = change_class_labels(y)
    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)

    fs_alg_names = ['MLP']#, 'SCA', 'DBN', 'TSFS']# ['TSFS']# ['TSFS']#
    num_feats = [0, 30]#, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] # 0 for the whole feature set

    cv = StratifiedKFold(n_splits=2, random_state=random_state, shuffle=False)
    results = dict()
    fold_idx = 0
    for train_index, test_index in cv.split(X, y):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        fold_dict = dict()
        ranked_feats = dict()
        for alg_name in fs_alg_names:
            new_vals = dfs(alg_name, X_train, y_train, feature_names)
            for key in new_vals.keys():# join two dictionary
                print('alg_name1111: ', key)
                ranked_feats[key] = new_vals[key]

        # now evaluate the results according to the number of selected features
        for alg_name in ranked_feats.keys():
            print('alg_name2222: ', alg_name)
            alg_dict = dict()

            for num_feat in num_feats:
                if len(ranked_feats[alg_name]) < num_feat:
                    alg_dict[num_feat] = {}
                    continue
                selected_feats = ranked_feats[alg_name] if num_feat == 0 else ranked_feats[alg_name][:num_feat]
                scores = all_clf_evaluator(X_train, X_test, y_train, y_test, feature_names, selected_feats)
                alg_dict[num_feat] = scores

            fold_dict[alg_name] = alg_dict

        results[fold_idx] = fold_dict
        fold_idx += 1 # update the fold index

        print('finish calculating results')
        print('Start writing results to ', dir_out, ' ....')

    write_results(results, dir_out)
    print('finish writing result')
    # return results

def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir

def write_results(results, dir_out):
    dir_out = standardize_dir(dir_out)
    writers = dict()
    print('results: ', results)
    # write results according to feature selection algorithm, each fs alg in a separate file
    for alg in results[0].keys():
        print('alg333333: ', alg)
        alg_out_path = dir_out + str(alg) + '.csv'
        writers[alg] = open(alg_out_path, 'w')
        writers[alg].write('alg_name,fold_id,num_feat,clf_name,acc,pre,rec,f1\n')
    for fold_idx in results.keys():
        fold_results = results[fold_idx]
        for alg in fold_results.keys():
            alg_fold_res = fold_results[alg]
            for num_feat in alg_fold_res.keys():
                feat_res = alg_fold_res[num_feat]
                if feat_res == {}:
                    continue
                for clf_name in feat_res.keys():
                    writers[alg].write(str(alg) + ',' + str(fold_idx) + ',' + str(num_feat) + ',' + str(clf_name))
                    for i_tmp in feat_res[clf_name]:
                        writers[alg].write(',' + str(i_tmp))
                    writers[alg].write('\n')

    # close all the writers
    for alg in writers.keys():
        writers[alg].close()

        # fold_dir = dir_out + str(fold_idx) + '/'
        # if not os.path.exists(fold_dir):
        #     os.mkdir(fold_dir)


def cross_validate_dir(dir_in, dir_out, normalized = False):
    files = os.listdir(dir_in)
    dir_in = standardize_dir(dir_in)
    dir_out = standardize_dir(dir_out)
    print('files: ', files)
    for file in files:
        if not file.endswith('.csv') or file == 'all_csv_stats.csv' or file == 'data_complexity.csv':
            continue
        file_name = file.replace('.csv', '')
        # create the directory for data file
        file_res_dir = dir_out + file_name + '/'
        if not os.path.exists(file_res_dir):
            os.mkdir(file_res_dir)
            cross_validate(dir_in + file, file_res_dir, normalize=normalized)



dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_Jun30/'
csv_path = dir + 'GLIOMA.csv'
dir_out = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/cross_validate_res/'
# dir = '/home/dong/bachelor-thesis-fs/data/standardize/'
# dir = '/home/dong/bachelor-thesis-fs/data/cross_validate_res/'
dir = '/home/dong/feature_selection/data/subset_June30/'
dir_out = '/home/dong/feature_selection/data/dl/'
dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/test/'
dir_out = dir
cross_validate_dir(dir, dir_out, True)
# cross_validate_dir(dir, dir_out + 'normalize/', True)
# cross_validate_dir(dir, dir_out + 'not_normalize/', False)
# cross_validate(csv_path, dir_out)