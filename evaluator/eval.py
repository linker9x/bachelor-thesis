import pandas as pd
import numpy as np
from time import time

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from dl_methods.methods.TSFS.utils import evaluate_reconstruction

# classifier_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#                     "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#                     "Naive Bayes", "QDA"]
#
# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]

classifier_names = ["Nearest Neighbors", "Linear SVM",
         "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()
    ]


ba_classifier_names = ['Linear SVM', 'Random Forest', 'Neural Net']


ba_classifiers = [
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=4),
    MLPClassifier(alpha=1, max_iter=1000)
    ]


def ba_clf_evaluator(X_train_full, X_test_full, y_train, y_test, feat_names, selected_feats):
    scores = dict()

    df_X_train = pd.DataFrame(X_train_full, columns=feat_names)
    # add evaluation result here
    df_X_test = pd.DataFrame(X_test_full, columns=feat_names)
    y_train = pd.DataFrame(y_train, columns=['class'])
    y_test = pd.DataFrame(y_test, columns=['class'])

    X_train = df_X_train[selected_feats]
    X_test = df_X_test[selected_feats]

    for classifier_name, clf in zip(ba_classifier_names, ba_classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        tmp_accuracy = clf.score(X_test, y_test)
        tmp_precision, tmp_recall, tmp_fscore, support = precision_recall_fscore_support(y_test,
                                                                                         y_pred, average=None)
        tmp_w_precision, tmp_w_recall, tmp_w_fscore, support = precision_recall_fscore_support(y_test,
                                                                                         y_pred, average='weighted')
        if classifier_name not in scores:
            scores[classifier_name] = [tmp_accuracy, tmp_w_precision, tmp_w_recall, tmp_w_fscore,
                                       tmp_precision, tmp_recall, tmp_fscore]

    return scores


def rf_evaluator(train_filepath, test_filepath, feature_selections, cv=False, add_Org=False):
    scores = []

    # load test and train data
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    # return index of class
    class_ind = train_df.columns[len(train_df.columns) - 1]

    # split training data into features and target
    train_features = train_df.drop(class_ind, axis=1)
    y_train = train_df[class_ind]

    # split test data into features and target
    test_features = test_df.drop(class_ind, axis=1)
    y_test = test_df[class_ind]

    # feature names
    feature_names = np.array(train_features.columns.values)

    # add original set of features to set
    if add_Org:
        feature_selections.append(('Original', []))

    for feature in feature_selections:
        # name of method
        name = feature[0]
        # name of features to be selected
        selections = feature[1]

        if name == 'Original':
            # all features
            X_train = train_features
            X_test = test_features

            # remove original afterwards
            feature_selections.pop()
        else:
            # features selected by the algorithm
            X_train = train_features[selections]
            X_test = test_features[selections]

        # CROSS VALIDATION
        if cv:
            # RANDOM GRID

            # values to assign to random grid
            n_estimators = [int(x) for x in np.linspace(100, 10000, num=5)]
            max_depth = [int(x) for x in np.linspace(1, 100, num=5)]
            max_depth.append(None)
            min_samples_split = [int(x) for x in np.linspace(2, 20, num=5)]
            min_samples_leaf = [int(x) for x in np.linspace(1, 10, num=5)]
            bootstrap = [True, False]

            # assembly of random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}

            # init and fit random forest and randomized CV
            print(name + ' starting Random Search')

            # build classifier
            r_rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=4)

            # run random search
            rf_random_CV = RandomizedSearchCV(estimator=r_rf, param_distributions=random_grid, n_iter=20, cv=3,
                                              random_state=42, verbose=2)
            start = time()
            rf_random_CV.fit(X_train, y_train)

            print('%s end Random Search. Time: %.2f seconds' % (name, time()-start))
            print('Best Params:' + rf_random_CV.best_params_)

            # PROCESS RESULT

            # best result from random search
            rand_result = rf_random_CV.best_params_

            # 3 values within range of best value
            n_est_val = int(rand_result.get('n_estimators'))
            n_estimators = [int(x) for x in np.linspace(n_est_val - (n_est_val/2),
                                                        n_est_val + (n_est_val/2), num=3)]

            # 3 values within range of best value
            min_split_val = int(rand_result.get('min_samples_split'))
            min_samples_split = [int(x) for x in np.linspace(min_split_val - (min_split_val/2) + 1,
                                                             min_split_val + (min_split_val/2), num=3)]

            # 3 values within range of best value
            min_leaf_val = int(rand_result.get('min_samples_leaf'))
            min_samples_leaf = [int(x) for x in np.linspace(min_leaf_val,
                                                            min_leaf_val + (min_leaf_val/2), num=3)]

            # if best result for max_depth is None OR int value
            if rand_result.get('max_depth') is None:
                max_depth = [rand_result.get('max_depth')]
            else:
                max_dep_val = int(rand_result.get('max_depth'))
                max_depth = [int(x) for x in np.linspace(max_dep_val,
                                                         max_dep_val + (max_dep_val/2), num=3)]

            # bootstrap is either true or false
            bootstrap = [rand_result.get('bootstrap')]

            # GRID SEARCH

            # create grid search based of result
            param_grid = {
                'bootstrap': bootstrap,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split,
                'n_estimators': n_estimators
            }

            # init and fit random forest and grid search
            print(name + ' starting Grid Search')

            # build classifier
            gs_rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=4)

            # run grid search
            gs = GridSearchCV(estimator=gs_rf, param_grid=param_grid, cv=3, verbose=2)

            start = time()
            gs.fit(X_train, y_train)
            print('%s end Grid Search. Time: %.2f seconds' % (name, time() - start))

            # final classifier
            rf = gs.best_estimator_

            # predict using test data
            y_pred = rf.predict(X_test)

        else:
            # WITHOUT CV
            rf = RandomForestClassifier(n_estimators=10000, random_state=42, n_jobs=4)

            # train the model
            rf.fit(X_train, y_train)

            # END REMOVE
            y_pred = rf.predict(X_test)

        #confusion matrix
        #labels = rf.classes_
        #cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=labels, index=labels)
        #plt.figure(figsize=(8, 5))
        #heatmap = sn.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")
        #heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        #heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        #plt.ylabel(name + ' true label')
        #plt.xlabel(name + ' predicted label')
        #plt.savefig('../images/' + name + '_cm.png', bbox_inches='tight')

        # calculate performance scores
        tmp_accuracy = rf.score(X_test, y_test)
        tmp_precision, tmp_recall, tmp_fscore, support = precision_recall_fscore_support(y_test,
                                                                                         y_pred, average=None)

        scores.append(('rf_' + name, tmp_accuracy, tmp_precision, tmp_recall, tmp_fscore))
        print(X_train.shape)
        print(X_test.shape)
        print('%s done' % name)

    return feature_selections, scores


def svm_evaluator(train_filepath, test_filepath, feature_selections, cv=False, add_Org=False):
    scores = []

    # load test and train data
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    # return index of class
    class_ind = train_df.columns[len(train_df.columns) - 1]

    # split training data into features and target
    train_features = train_df.drop(class_ind, axis=1)
    y_train = train_df[class_ind]

    # split test data into features and target
    test_features = test_df.drop(class_ind, axis=1)
    y_test = test_df[class_ind]

    # feature names
    feature_names = np.array(train_features.columns.values)

    # add original set of features to set
    if add_Org:
        feature_selections.append(('Original', []))

    for feature in feature_selections:
        # name of method
        name = feature[0]
        # name of features to be selected
        selections = feature[1]

        if name == 'Original':
            # all features
            X_train = train_features
            X_test = test_features

            # remove original afterwards
            feature_selections.pop()
        else:
            # features selected by the algorithm
            X_train = train_features[selections]
            X_test = test_features[selections]

        # CROSS VALIDATION
        if cv:
            # RANDOM GRID

            # values to assign to random grid
            kernel = ['linear', 'rbf']
            C = [1, 10, 100, 1000]
            gamma = [0.01, 0.001, 0.0001, 'auto']
            decision_function_shape = ['ovo', 'ovr']
            shrinking = [True, False]

            # assembly of random grid
            random_grid = {'kernel': kernel,
                           'C': C,
                           'gamma': gamma,
                           'decision_function_shape': decision_function_shape,
                           'shrinking': shrinking}

            # init and fit random forest and randomized CV
            print(name + ' starting Random Search')

            # build classifier
            r_svm = SVC(random_state=42)

            # run random search
            svm_random_CV = RandomizedSearchCV(estimator=r_svm, param_distributions=random_grid, n_iter=20, cv=3,
                                              random_state=42, verbose=2)
            start = time()
            svm_random_CV.fit(X_train, y_train)

            print('%s end Random Search. Time: %.2f seconds' % (name, time()-start))
            print('Best Params:' + svm_random_CV.best_params_)

            # PROCESS RESULT

            # best result from random search
            rand_result = svm_random_CV.best_params_

            # best value for kernel
            kernel = [rand_result.get('kernel')]
            gamma = [rand_result.get('gamma')]
            decision_function_shape = [rand_result.get('decision_function_shape')]
            shrinking = [rand_result.get('shrinking')]

            # 3 values within range of best value
            C_val = int(rand_result.get('C'))
            C = [int(x) for x in np.linspace(C_val - (C_val/2) + 1, C_val + (C_val/2), num=3)]

            # GRID SEARCH

            # create grid search based of result
            param_grid = {'kernel': kernel,
                           'C': C,
                           'gamma': gamma,
                           'decision_function_shape': decision_function_shape,
                           'shrinking': shrinking}

            # init and fit random forest and grid search
            print(name + ' starting Grid Search')

            # build classifier
            gs_svc = SVC(random_state=42)

            # run grid search
            gs = GridSearchCV(estimator=gs_svc, param_grid=param_grid, cv=3, verbose=2)

            start = time()
            gs.fit(X_train, y_train)
            print('%s end Grid Search. Time: %.2f seconds' % (name, time() - start))

            # final classifier
            svm = gs.best_estimator_

            # predict using test data
            y_pred = svm.predict(X_test)

        else:
            # WITHOUT CV
            svm = SVC(random_state=42)

            # train the model
            svm.fit(X_train, y_train)

            # END REMOVE
            y_pred = svm.predict(X_test)

        #confusion matrix
        #labels = rf.classes_
        #cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=labels, index=labels)
        #plt.figure(figsize=(8, 5))
        #heatmap = sn.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")
        #heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        #heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        #plt.ylabel(name + ' true label')
        #plt.xlabel(name + ' predicted label')
        #plt.savefig('../images/' + name + '_cm.png', bbox_inches='tight')

        # calculate performance scores
        tmp_accuracy = svm.score(X_test, y_test)
        tmp_precision, tmp_recall, tmp_fscore, support = precision_recall_fscore_support(y_test,
                                                                                         y_pred, average=None)

        scores.append(('svm_' + name, tmp_accuracy, tmp_precision, tmp_recall, tmp_fscore))
        print(X_train.shape)
        print(X_test.shape)
        print('%s done' % name)

    return feature_selections, scores


def all_clf_evaluator(train_features_val, test_features_val, y_train, y_test, feat_names, selected_feats):
    scores = dict()
    train_features = pd.DataFrame(train_features_val, columns=feat_names)
    # add evaluation result here
    test_features = pd.DataFrame(test_features_val, columns=feat_names)
    y_train = pd.DataFrame(y_train, columns=['class'])
    y_test = pd.DataFrame(y_test, columns=['class'])

    X_train = train_features[selected_feats]
    X_test = test_features[selected_feats]
    scores['reconstruction'] = evaluate_reconstruction(train_features_val, X_train.values, test_features_val, X_test.values, len(selected_feats))

    for classifier_name, clf in zip(classifier_names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        tmp_accuracy = clf.score(X_test, y_test)
        tmp_precision, tmp_recall, tmp_fscore, support = precision_recall_fscore_support(y_test,
                                                                                         y_pred, average='weighted')
        if classifier_name not in scores:
            scores[classifier_name] = [tmp_accuracy, tmp_precision, tmp_recall,
                                        tmp_fscore]

    return scores

