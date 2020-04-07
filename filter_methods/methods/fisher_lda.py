from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as FisherLDA
import pandas as pd


def fisher_lda(X, y, feat_names):

    lda = FisherLDA()
    lda.fit_transform(X, y)

    results = sorted(zip(map(lambda x: round(x, 2), lda.coef_[0]), feat_names), reverse=True)

    return [x[1] for x in results]
