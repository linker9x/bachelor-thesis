from sklearn.linear_model import ElasticNet, ElasticNetCV


def elastic_net(X, y, feat_names):
    enetcv = ElasticNetCV(n_alphas=20, l1_ratio=[.1, .3, .5, .7, .9, .95, 1], cv=10, max_iter=100000,
                          normalize=True, n_jobs=4)
    enetcv.fit(X, y)

    enet = ElasticNet(alpha=enetcv.alpha_, l1_ratio=enetcv.l1_ratio_, max_iter=10000, normalize=True)
    enet.fit(X, y)

    # print("EN_Alpha=", enetcv.alpha_)
    # print("EN_L1 ratio=", enetcv.l1_ratio_)

    results = sorted(zip(map(lambda x: round(x, 4), abs(enet.coef_)), feat_names), reverse=True)

    return [x[1] for x in results if x[0] > 0.0000]
