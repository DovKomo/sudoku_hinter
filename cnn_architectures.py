from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from tensorflow.keras import regularizers


def cnn_architecture_1(num_classes, input_shape=(28, 28, 1)):
    """Creates a CNN architecture."""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, input_shape=input_shape, padding='same', activation='relu'))
    model.add(MaxPooling2D())  # pool_size=(2, 2)
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def cnn_architecture_2(num_classes, input_shape=(28, 28, 1)):
    """Creates a CNN architecture (takes longer time)."""
    # https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist/notebook
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())  # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def cnn_architecture_3(num_classes, input_shape=(28, 28, 1)):
    """https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392."""
    model = Sequential()

    model.add(Conv2D(32, kernel_size=5, strides=1, activation='relu', input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(0.0005)))

    model.add(Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dense(84, activation='relu', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def cnn_architecture_4(num_classes, input_shape=(28, 28, 1)):
    """https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1."""
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))
    return model


def cnn_architecture_5(num_classes, input_shape=(28, 28, 1)):
    """Creates a CNN architecture (takes longer time)."""
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # added myself
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))  # added myself
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def cnn_architecture_6(num_classes, input_shape=(28, 28, 1)):
    """Creates a CNN architecture (takes longer time)."""
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # added myself
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def cnn_architecture_7(num_classes, input_shape=(28, 28, 1)):
    """https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6."""
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def cnn_architecture_8(num_classes, input_shape=(28, 28, 1)):
    """https://www.kaggle.com/blurredmachine/mnist-classification-eda-pca-cnn-99-7-score."""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    return model
