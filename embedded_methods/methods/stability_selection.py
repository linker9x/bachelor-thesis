from sklearn.linear_model import RandomizedLasso
import warnings
warnings.filterwarnings('ignore')


def stability_selection(X, y, feat_names):
    randlasso = RandomizedLasso()

    randlasso.fit(X, y)

    results = sorted(zip(map(lambda x: round(x, 4), randlasso.scores_), feat_names), reverse=True)

    return [x[1] for x in results if x[0] > 0.0000]
