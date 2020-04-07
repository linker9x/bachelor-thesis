import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import zscore


import numpy as np

from dl_methods.dl_wrapper import *
from evaluator.eval import all_clf_evaluator
from dl_methods.methods.dbn_mlp_sca.deep_learning.classification import change_class_labels
import os

def cross_validate(csv_path, dir_out, file_name, random_state=42):
    df = pd.read_csv(csv_path)
    df['class'] = pd.factorize(df['class'])[0] + 1
    y = df.pop('class').values
    df = df.apply(zscore)
    feature_names = np.array(df.columns.values)
    X = df.values
    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)
    _, y = change_class_labels(y)
    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)

    fs_alg_names = ['MLP']#['TSFS','MLP']# ['TSFS']#
    # num_feats = [0, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] # 0 for the whole feature set
    # per_feats = [i * 1.0 for i in range(20)]
    cv = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=False)
    fold_idx = 0
    for train_index, test_index in cv.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        train_df_X = pd.DataFrame(X_train, columns=feature_names)
        train_df_y = pd.DataFrame(y_train, columns=['class'])
        test_df_X = pd.DataFrame(X_test, columns=feature_names)
        test_df_y = pd.DataFrame(y_test, columns=['class'])
        train_df = pd.concat([train_df_X, train_df_y], axis=1)
        test_df = pd.concat([test_df_X, test_df_y], axis=1)
        train_fold_path = dir_out + file_name + '_train_' + str(fold_idx) + '.csv'
        test_fold_path = dir_out + file_name + '_test_' + str(fold_idx) + '.csv'
        train_df.to_csv(train_fold_path, index=False)
        test_df.to_csv(test_fold_path, index=False)
        fold_idx += 1


    print('finish writing result')
    # return results

def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir


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
            cross_validate(dir_in + file, file_res_dir, file_name)



dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/'
csv_path = dir + 'GLIOMA.csv'
# dir_out = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/cross_validate_res/'
# dir_out = dir + 'starting_points/'
# dir = '/home/dong/feature_selection/data/subset_June30/'
# dir_out = '/home/dong/feature_selection/data/percentage/'
# dir = '/home/dong/bachelor-thesis-fs/data/standardize/'
# dir = '/home/dong/bachelor-thesis-fs/data/cross_validate_res/'
dir_out = dir + 'fold_data/'
cross_validate_dir(dir, dir_out, True)
# cross_validate_dir(dir, dir_out + 'normalize/', True)
# cross_validate(csv_path, dir_out)