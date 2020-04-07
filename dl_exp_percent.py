import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import zscore


import numpy as np

from dl_methods.dl_wrapper import *
from evaluator.eval import all_clf_evaluator
from dl_methods.methods.dbn_mlp_sca.deep_learning.classification import change_class_labels
import os
from stability.similarity import *


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

    fs_alg_names = ['MLP','TSFS']# ['TSFS']#
    # num_feats = [0, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] # 0 for the whole feature set
    per_feats = [i * 1.0 for i in range(20)]
    # per_feats = [0]
    # per_feats = [1.0, 2.0]

    cv = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=False)
    results = dict()
    fold_idx = 0
    fold_returned_dict = dict()
    selected_feats_dict = dict()
    for train_index, test_index in cv.split(X, y):
        print(fold_idx, " Train Index: ", train_index, "\n")
        print(fold_idx, " Test Index: ", test_index)

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
            no_feats = len(feature_names)
            for per_feat in per_feats:
                selected_feats = feature_names if per_feat == 0 else ranked_feats[alg_name][:int(per_feat * no_feats / 100) + 1]

                if per_feat == per_feats[-1]:
                    if alg_name not in selected_feats_dict:
                        selected_feats_dict[alg_name] = dict()
                    if fold_idx not in selected_feats_dict[alg_name]:
                        selected_feats_dict[alg_name][fold_idx] = selected_feats

                if alg_name not in fold_returned_dict:
                    fold_returned_dict[alg_name] = dict()
                if per_feat not in fold_returned_dict[alg_name]:
                    fold_returned_dict[alg_name][per_feat] = list()
                fold_returned_dict[alg_name][per_feat].append(selected_feats)

                scores = all_clf_evaluator(X_train, X_test, y_train, y_test, feature_names, selected_feats)
                alg_dict[per_feat] = scores

            fold_dict[alg_name] = alg_dict

        results[fold_idx] = fold_dict
        fold_idx += 1 # update the fold index

    print('finish calculating results')
    print('Start writing results to ', dir_out, ' ....')

    # print('fold return dict: ', fold_returned_dict)
    # Update scability index in the results dict
    every_fold_scores = dict()
    for alg_name in fold_returned_dict.keys():
        if alg_name not in every_fold_scores:
            every_fold_scores[alg_name] = dict()
        for per_feat in fold_returned_dict[alg_name].keys():
            every_fold_scores[alg_name][per_feat] = get_smilarity_scores(fold_returned_dict[alg_name][per_feat], len(feature_names))

    # Update retured results, every fold have the same similarity scores
    for fold_idx in results.keys():
        for alg_name in every_fold_scores.keys():
            for per_feat in every_fold_scores[alg_name].keys():
                for key in every_fold_scores[alg_name][per_feat].keys():
                    results[fold_idx][alg_name][per_feat][key] = every_fold_scores[alg_name][per_feat][key]

    write_results(results, dir_out, selected_feats_dict)
    print('finish writing result')
    # return results

def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir

def write_results(results, dir_out, selected_feats_dict):
    dir_out = standardize_dir(dir_out)
    writers = dict()
    # print('results: ', results)
    # write results according to feature selection algorithm, each fs alg in a separate file
    for alg in results[0].keys():
        print('alg333333: ', alg)
        alg_out_path = dir_out + str(alg) + '.csv'
        writers[alg] = open(alg_out_path, 'w')
        writers[alg].write('alg_name,fold_id,num_feat,clf_name,acc,pre,rec,f1,' +
                           'tanimoto,tanimoto_std,hamming,hamming_std,kuncheva,kuncheva_std,jaccard,jaccard_std\n') # add similarity header
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

    print(selected_feats_dict)

    for alg_name in selected_feats_dict.keys():
        selected_writer_path = dir_out + alg_name + '_selected_feat.csv'
        selected_writer = open(selected_writer_path, 'w')
        for fold_idx in range(5):
            selected_writer.write(','.join(selected_feats_dict[alg_name][fold_idx]) + '\n')
        selected_writer.close()

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



dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/'
csv_path = dir + 'GLIOMA.csv'
dir_out = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/cross_validate_res/'
dir_out = dir + 'starting_points/'
# dir = '/home/dong/feature_selection/data/subset_June30/'
# dir_out = '/home/dong/feature_selection/data/percentage/'
# dir = '/home/dong/bachelor-thesis-fs/data/standardize/'
# dir = '/home/dong/bachelor-thesis-fs/data/cross_validate_res/'
# cross_validate_dir(dir, dir_out, True)
# cross_validate_dir(dir, dir_out + 'normalize/', True)
# cross_validate(csv_path, dir_out)


test_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/test/'
dir_out = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/jul17/'
cross_validate_dir(dir, dir_out, True)