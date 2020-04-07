# -*- coding: utf-8 -*-
import dl_methods.methods.dbn_mlp_sca.deep_learning.deep_feat_select_mlp as mlp
import dl_methods.methods.dbn_mlp_sca.deep_learning.deep_feat_select_DBN as dbn
import dl_methods.methods.dbn_mlp_sca.deep_learning.deep_feat_select_ScA as sca
from dl_methods.methods.conc_autoenc import CAE
import dl_methods.methods.TSFS.methods as method_functions


from gc import collect as gc_collect


support_names = ['TSFS', 'MLP', 'SCA', 'DBN', 'CAE_S', 'CAE_U']
# All return a list of ranked feature names
def dfs(alg_name, x_train, y_train, feat_names, num_feats=None):
    if alg_name not in support_names:
        print(alg_name, ' is not a supported deep learning algorithm!!!!!')
        return None

    if alg_name in ['CAE_S', 'CAE_U']:
        print('Working with ', alg_name, 'algorithm')

        if not num_feats:
            num_feats = 20

        if alg_name == 'CAE_S':
            results = CAE(x_train, y_train, feat_names, num_feats, sup=True)

        if alg_name == 'CAE_U':
            results = CAE(x_train, y_train, feat_names, num_feats, sup=False)

        return {alg_name : results}


    if alg_name == 'MLP':
        print('Working with ', alg_name, 'algorithm')
        classifier, training_time = mlp.train_model(train_set_x_org=x_train,
                                                    train_set_y_org=y_train,
                                                    valid_set_x_org=x_train,
                                                    valid_set_y_org=y_train
                                                    )
        # the scores/ weights
        feat_weights = classifier.params[0].get_value()

        # get the list of (feat_weight, feat_name) in descending order according to the feat_weight.
        results = sorted(zip(map(lambda x: round(x, 4), feat_weights), feat_names), reverse=True)

        gc_collect()
        return {alg_name : [x[1] for x in results]}

    if alg_name == 'SCA':
        print('Working with ', alg_name, 'algorithm')
        classifier, training_time = sca.train_model(train_set_x_org=x_train,
                                                    train_set_y_org=y_train,
                                                    valid_set_x_org=x_train,
                                                    valid_set_y_org=y_train
                                                    )
        # the scores/ weights
        feat_weights = classifier.params[0].get_value()

        # get the list of (feat_weight, feat_name) in descending order according to the feat_weight.
        results = sorted(zip(map(lambda x: round(x, 4), feat_weights), feat_names), reverse=True)

        gc_collect()
        return {alg_name : [x[1] for x in results]}

    if alg_name == 'DBN':
        print('Working with ', alg_name, 'algorithm')
        classifier, training_time = dbn.train_model(train_set_x_org=x_train,
                                                    train_set_y_org=y_train,
                                                    valid_set_x_org=x_train,
                                                    valid_set_y_org=y_train
                                                    )
        # the scores/ weights
        feat_weights = classifier.params[0].get_value()

        # get the list of (feat_weight, feat_name) in descending order according to the feat_weight.
        results = sorted(zip(map(lambda x: round(x, 4), feat_weights), feat_names), reverse=True)

        gc_collect()
        return {alg_name : [x[1] for x in results]}

    if alg_name == 'TSFS':
        # MCFS: Unsupervised feature selection for multi-cluster data
        # my_se: Representation Reconstruction Feature Selection with SpectralEmbedding
        # my_mds: Representation Reconstruction Feature Selection with MDS (Multidimensional scaling)
        # my_lle: Representation Reconstruction Feature Selection with LocallyLinearEmbedding
        # my_tsne: Representation Reconstruction Feature Selection with t-distributed Stochastic Neighbor Embedding.
        # my_isomap: Representation Reconstruction Feature Selection with Isomap Embedding
        # my_autoencoder: Representation Reconstruction Feature Selection with auto encoder
        tsfs_methods = ['MCFS', 'my_mds'] # ['my_se', 'my_autoencoder']#
        # ['my_tsne', 'my_lle', 'my_isomap', 'MCFS', 'aefs', 'my_se', 'my_mds', 'my_autoencoder']
        results = dict()
        for method in tsfs_methods:
            print('Working on TSFS ', method)
            indexes = getattr(method_functions, method)(x_train.astype(float), y_train)
            results['TSFS_' + method] = [feat_names[i] for i in indexes]

        gc_collect()
        return results