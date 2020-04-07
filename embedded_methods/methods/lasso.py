from sklearn.linear_model import Lasso, LassoCV


def lasso(X, y, feat_names):
    lassocv = LassoCV(n_alphas=20, cv=10, max_iter=100000, normalize=True, n_jobs=4)
    lassocv.fit(X, y)

    lasso = Lasso(alpha=lassocv.alpha_, max_iter=10000, normalize=True)
    lasso.fit(X, y)

    # print("LAS_Alpha=", lassocv.alpha_)

    results = sorted(zip(map(lambda x: round(x, 4), abs(lasso.coef_)), feat_names), reverse=True)

    return [x[1] for x in results if x[0] > 0.0000]
