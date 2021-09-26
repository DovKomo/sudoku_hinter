import os

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils


def plot_mnist_images(nrows=2, ncols=3, show=False):
    """Plots mnist hand written digits."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # load 28×28 sized images
    print(f'X_train shape: {X_train.shape}', f'X_test shape: {X_test.shape}')
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    for ax, i in zip(axes.flatten(), range(len(axes.flatten()))):
        ax.set_axis_off()
        ax.imshow(X_train[i], cmap=plt.get_cmap('gray'))
        ax.set_title(f'{y_train[i]}')
    plt.savefig('outputs//digit_recognition//mnist_images.png')
    if show:
        plt.show()


def train_cnn_network(epochs=10, batch_size=200, shuffle=True, show=False):
    """Performs training steps."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # load data
    # reshape to be [samples][width][height][channels]
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    # build the model
    model = cnn_architecture(num_classes, input_shape=(28, 28, 1))
    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
                        shuffle=shuffle)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    plot_model_performance(history, final_test_acc=scores[1], final_test_loss=scores[0], show=show)

    # save model:
    model.save('outputs//digit_recognition//saved_model')


def cnn_architecture(num_classes, input_shape=(28, 28, 1)):
    """Creates a CNN architecture."""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D())  # pool_size=(2, 2)
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_model_performance(history, final_test_acc, final_test_loss, show=False):
    """Plots the model performance (accuracy and loss) in each epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, final in zip([ax1, ax2], ['accuracy', 'loss'], [final_test_acc, final_test_loss]):
        ax.plot(history.history[f'{metric}'])
        ax.plot(history.history[f'val_{metric}'])
        ax.set_title(f'model {metric}: {round(final * 100, 2)}%')
        ax.set(xlabel='epoch', ylabel=f'{metric}')
    plt.legend(['train', 'val'], loc='upper left')

    img_file = 'outputs//digit_recognition//model_performance.png'
    if os.path.exists(img_file):
        os.remove(img_file)
    plt.savefig(img_file)

    if show:
        plt.show()


if __name__ == "__main__":
    plot_mnist_images(nrows=3, ncols=5, show=False)
    train_cnn_network(epochs=10, batch_size=200, shuffle=True, show=False)
