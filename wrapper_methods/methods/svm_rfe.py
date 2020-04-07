from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC


def svm_rfe(X, y, feat_names):
    svm = LinearSVC(max_iter=10000)
    rfecv = RFECV(estimator=svm, cv=5, step=2, scoring='accuracy', n_jobs=4)

    rfecv.fit(X, y)

    results = sorted(zip(rfecv.ranking_, feat_names), reverse=False)

    return [x[1] for x in results]