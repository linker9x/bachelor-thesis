from pyHSICLasso import HSICLasso

def hsic(num_features, hsic_data, method='regression'):
    hsic_lasso = HSICLasso()
    hsic_lasso.input(hsic_data)

    if method == 'regression':
        hsic_lasso.regression(num_features)
    else:
        hsic_lasso.classification(num_features)

    return hsic_lasso.get_features()
