from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np


def chi_squared(X, y, feature_names, num_features):
    kbest = SelectKBest(score_func=chi2, k=num_features)
    kbest.fit_transform(np.absolute(X), y)

    index_vals = np.where(kbest.get_support())
    results = [feature_names[x] for x in index_vals[0]]

    return results