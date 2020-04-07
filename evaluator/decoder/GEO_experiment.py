import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../..')))
from os.path import join

import numpy as np
import tensorflow as tf
import pandas as pd
from numpy.random import RandomState

from sklearn.model_selection import train_test_split
from evaluator.decoder.concrete_estimator import run_experiment

dataset_dir = 'datasets'


def load_GEO():
    df_train = pd.read_csv('../../data/Leukemia_train.csv')
    test = pd.read_csv('../../data/Leukemia_test.csv')

    # split training data into training/validation sets
    rng = RandomState(42)
    train = df_train.sample(frac=.8, random_state=rng)
    val = df_train.loc[~df_train.index.isin(train.index)]

    # class col index number
    class_ind = df_train.columns[len(df_train.columns) - 1]

    # training set
    train_X = train.drop(class_ind, axis=1).values
    train_y = train[class_ind].values

    # validation set
    val_X = val.drop(class_ind, axis=1).values
    val_y = val[class_ind].values

    # test set
    test_X = test.drop(class_ind, axis=1).values
    test_y = test[class_ind].values

    return (train_X, train_X), (val_X, val_X), (test_X, test_X)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    train, val, test = load_GEO()

    sz = 9000

    for i in range(3):
        probabilities = run_experiment('%d_GEO_linear' % i, train, val, test, 943, [], 5000, 256, 0.0003, 0.1)
        indices = np.argmax(probabilities, axis=1)
        print(indices)
        # for hidden_units in ([], [sz], [sz, sz], [sz, sz, sz]):
        #    test_GEO('%d_GEO_%d_hidden_layers' % (i, len(hidden_units)), indices, hidden_units)

    '''
    for j, i in enumerate([900, 850, 800, 750, 700, 650, 600]):
        if j < 2 or j % 2 == 0:
            continue
        probabilities = run_experiment('%d_GEO_%d' % (j, i), train, val, test, i, [], 5000, 256, 0.0003, 0.1)
        indices = np.argmax(probabilities, axis = 1)
        print(indices)
        test_GEO('%d_GEO_%d_genes_selected' % (j, i), indices, [])
    '''


if __name__ == '__main__':
    main()