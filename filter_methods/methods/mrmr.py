from skfeature.function.information_theoretical_based import MRMR
import numpy as np 


def mrmr(X, y, feat_names, num_features):
    indexes,_,_ = MRMR.mrmr(X, np.ravel(y), n_selected_features=num_features)
    results = [feat_names[idx] for idx in indexes]

    return results
