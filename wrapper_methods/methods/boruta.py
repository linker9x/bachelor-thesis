from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def boruta_fs(X, y, feat_names):
    rfc = RandomForestClassifier(n_estimators=10000, n_jobs=4, max_depth=1)
    boruta = BorutaPy(rfc, n_estimators='auto', verbose=2, max_iter=50)

    boruta.fit(X, y)

    results = sorted(zip(boruta.ranking_, feat_names), reverse=False)

    return [x[1] for x in results]