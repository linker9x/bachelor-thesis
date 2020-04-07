from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
import numpy as np
from sklearn.model_selection import train_test_split


def CAE(x_train, y_train, feature_names, num_features, sup=False, x_test=None, y_test=None):
    if None in [x_test, y_test]:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

    x_train = np.reshape(x_train, (len(x_train), -1))
    x_test = np.reshape(x_test, (len(x_test), -1))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    def decoder(x):
        x = Dense(320)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(320)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        if sup:
            x = Dense(y_test.shape[1])(x)  # This has to be the number of classes?
        else:
            x = Dense(len(feature_names))(x)  # This has to be the number of features?
        return x

    selector = ConcreteAutoencoderFeatureSelector(K=num_features, output_function=decoder,
                                                  tryout_limit=1, num_epochs=100)

    if sup:
        selector.fit(x_train, y_train, x_test, y_test)
    else:
        selector.fit(x_train, x_train, x_test, x_test)

    index_fs = selector.get_support(indices=True)
    names_fs = [feature_names[i] for i in index_fs]

    return names_fs
