import dl_methods.methods.TSFS.methods as method_functions



def TSFS(train_X, train_y, feature_names, method_name='my_tsne'):
    # method(s) to use in the algorithim
    methods = [method_name]


    # result path for each method
    for method in methods:
        results = {}
        dataset = 'cur'

        X = train_X  # data
        X = X.astype(float)
        y = train_y  # label

        # if there isn't a result for the dataset yet
        if (dataset not in results.keys()):
            # create a result
            results[dataset] = {}
            # apply the method
            idx = getattr(method_functions, method)(X, y, dataset=dataset)
            results[dataset]['feature_ranking'] = idx
        else:
            idx = results[dataset]['feature_ranking']

        index_fs = results[dataset]['feature_ranking']
        names_fs = [feature_names[i] for i in index_fs]

    return names_fs
