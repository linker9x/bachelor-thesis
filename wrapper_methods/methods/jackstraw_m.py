import numpy as np
from wrapper_methods.methods.jackstraw.jackstraw import Jackstraw


def jackstraw(X, feat_names):

    jackstraw = Jackstraw(alpha=0.05)

    jackstraw.fit(X, method='pca', rank=1)
    rejected = jackstraw.rejected
    rejected = [feat_names[x] for x in rejected]

    return np.setdiff1d(feat_names, rejected)