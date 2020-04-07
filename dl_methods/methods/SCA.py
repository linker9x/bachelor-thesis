import dl_methods.methods.dbn_mlp_sca.deep_learning.deep_feat_select_ScA as sca
from dl_methods.methods.dbn_mlp_sca.deep_learning.classification import change_class_labels
from sklearn.model_selection import train_test_split
from gc import collect as gc_collect


def SCA(X, y, feature_names):
    _, y = change_class_labels(y)
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.20, random_state=42)

    # train
    lambda1s = [0.012, 0.010]  # numpy.arange(0.0700,-0.001,-0.001)

    for i in range(len(lambda1s)):
        classifier, training_time = sca.train_model(train_set_x_org=train_X,
                                                    train_set_y_org=train_y,
                                                    valid_set_x_org=val_X,
                                                    valid_set_y_org=val_y)

        # the scores/ weights
        param0 = classifier.params[0].get_value()

        results = sorted(zip(map(lambda x: round(x, 4), param0), feature_names), reverse=True)

    gc_collect()
    return [x[1] for x in results]
