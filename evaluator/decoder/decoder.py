import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../..')))

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import RandomState

from datetime import datetime
from os.path import join, exists
from os import makedirs
import json

from evaluator.decoder.concrete_estimator import dataset_input_fn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

figure_dir = './dump/'


def load_DS(train_fp, test_fp):
    df_train = pd.read_csv(train_fp)
    test = pd.read_csv(test_fp)

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

    return train_X, val_X, test_X


def MSELR(indices, train, val, test):
    not_indices = list(set(range(train.shape[1])) - set(indices))
    LR = LinearRegression(n_jobs=6)
    LR.fit(train[:, indices], train[:, not_indices])
    mse_val = mean_squared_error(LR.predict(val[:, indices]), val[:, not_indices])
    mse_test = mean_squared_error(LR.predict(test[:, indices]), test[:, not_indices])
    return float(mse_val), float(mse_test)


def test_DS(name, indices, hidden_units, train_fp, test_fp):
    tf.logging.set_verbosity(tf.logging.INFO)

    train, val, test = load_DS(train_fp, test_fp)

    indices = list(indices)
    not_indices = list(set(range(train.shape[1])) - set(indices))

    print(list(indices))

    mse_val = 0
    mse_test = 0
    
    if len(hidden_units) == 0:
        mse_val, mse_test = MSELR(indices, train, val, test)
    else:
        num_epochs = 100
        batch_size = 256
        steps_per_epoch = (train.shape[0] + batch_size - 1) // batch_size
        epochs_per_evaluation = 50
        dropout = 0.1
        learning_rate = 1e-3
        # optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        optimizer = lambda: tf.train.AdamOptimizer(learning_rate=
                                                   tf.train.exponential_decay(learning_rate=learning_rate,
                                                                              global_step=tf.train.get_global_step(),
                                                                              decay_steps=steps_per_epoch,
                                                                              decay_rate=0.95, staircase=True))
        
        train_input_fn = lambda: dataset_input_fn((train[:, indices], train[:, not_indices]), batch_size, -1)
        eval_input_fn = lambda: dataset_input_fn((val[:, indices], val[:, not_indices]), batch_size, seed=1)
        test_input_fn = lambda: dataset_input_fn((test[:, indices], test[:, not_indices]), batch_size, seed=1)

        feature_columns = [tf.feature_column.numeric_column(key='features', shape=[len(indices)])]

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        regressor = tf.estimator.DNNRegressor(hidden_units=hidden_units, feature_columns=feature_columns,
                                              label_dimension=len(not_indices), optimizer=optimizer,
                                              loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
                                              activation_fn=tf.nn.leaky_relu, dropout=dropout, config=
                                              tf.estimator.RunConfig(save_checkpoints_steps=epochs_per_evaluation * steps_per_epoch,
                                                                     save_summary_steps=steps_per_epoch,
                                                                     log_step_count_steps=steps_per_epoch,
                                                                     session_config=session_config))

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_epochs * steps_per_epoch)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, throttle_secs=0)

        tf.estimator.train_and_evaluate(regressor, train_spec, eval_spec)

        eval_evaluation_result = regressor.evaluate(eval_input_fn)

        test_evaluation_result = regressor.evaluate(test_input_fn)
        
        mse_val = float(eval_evaluation_result['average_loss'])
        
        mse_test = float(test_evaluation_result['average_loss'])
    
    results = (mse_val, mse_test)
    print(name, results)
    with open(join(figure_dir, name), 'w') as f:
        json.dump(results, f)
        
    return results


def run_test(name, sz, idx, train_fp, test_fp):
    sz = sz
    cae_indices = idx
    for hidden_units in [[], [sz], [sz, sz], [sz, sz, sz]]:
        for i in range(3):
            #test_DS('%d_landmarks_%d_hidden_layers' % (i, len(hidden_units)), np.arange(943), hidden_units)
            test_DS('%s_%d_CAE_%d_hidden_layers' % (name, i, len(hidden_units)), cae_indices, hidden_units, train_fp, test_fp)

    return None
