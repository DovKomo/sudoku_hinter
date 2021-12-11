import os
import time

from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from cnn_architectures import *
from train_plots import *


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


def prepare_data(x, size=28):
    """Prepares images: reshapes and normalizes."""
    # reshape to be [samples][width][height][channels]
    x = x.reshape((x.shape[0], size, size, 1)).astype('float32')
    # normalize inputs from 0-255 to 0-1
    x = x / 255
    return x


def define_model(model_name, num_classes, img_size=28):
    # build the model
    if model_name == 'cnn_architecture_1':
        model = cnn_architecture_1(num_classes, input_shape=(img_size, img_size, 1))
    if model_name == 'cnn_architecture_2':
        model = cnn_architecture_2(num_classes, input_shape=(img_size, img_size, 1))
    if model_name == 'cnn_architecture_3':
        model = cnn_architecture_3(num_classes, input_shape=(img_size, img_size, 1))
    if model_name == 'cnn_architecture_4':
        model = cnn_architecture_4(num_classes, input_shape=(img_size, img_size, 1))
    if model_name == 'cnn_architecture_5':
        model = cnn_architecture_5(num_classes, input_shape=(img_size, img_size, 1))
    if model_name == 'cnn_architecture_6':
        model = cnn_architecture_6(num_classes, input_shape=(img_size, img_size, 1))
    if model_name == 'cnn_architecture_7':
        model = cnn_architecture_7(num_classes, input_shape=(img_size, img_size, 1))
    if model_name == 'cnn_architecture_8':
        model = cnn_architecture_8(num_classes, input_shape=(img_size, img_size, 1))
    return model


def train_cnn_network(epochs=10, batch_size=200, val_ratio=0.2, early_stop_epochs=5, img_size=28, learning_rate=0.001,
                      shuffle=True, model_name='cnn_architecture_2'):
    """Performs training steps."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # load data
    plot_data_distribution(y_train, y_test, model_name=model_name)
    X_train = prepare_data(X_train, size=img_size)
    X_test = prepare_data(X_test, size=img_size)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # ------------------------------
    # build the model
    model = define_model(model_name, num_classes, img_size)

    with open(f'outputs//digit_recognition//{model_name}//model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    # Fit the model
    checkpoint = ModelCheckpoint(f'outputs//digit_recognition//{model_name}//tmp//checkpoints',
                                 monitor='val_loss', save_best_only=True,
                                 save_weights_only=True, period=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_epochs)
    callbacks = [checkpoint, es]
    start = time.time()
    history = model.fit(X_train, y_train, validation_split=val_ratio, epochs=epochs, batch_size=batch_size,
                        shuffle=shuffle, callbacks=callbacks)
    end = time.time()
    # ------------------------------
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    plot_model_performance(history, final_test_acc=scores[1], final_test_loss=scores[0], model_name=model_name)

    # get predictions
    y_pred_raw = model.predict(X_test)
    y_pred = np.argmax(y_pred_raw, axis=1)  # take the largest probability
    y_test = np.argmax(y_test, axis=1)

    # classification report
    classes = list(np.unique(y_test))

    report = classification_report(y_test, y_pred, digits=3, target_names=classes, output_dict=True)
    classification_df = pd.DataFrame(report).transpose()

    classification_df.to_csv(f'outputs//digit_recognition//{model_name}//classification_report.csv')

    # confusion matrix
    get_confusion_matrix(y_pred, y_test, classes, model_name=model_name)

    # roc curve
    get_roc_curves(y_pred_raw, y_test, model_name=model_name)

    # save model:
    model.save(f'outputs//digit_recognition//{model_name}//saved_model.h5')
    return end - start, scores[1], scores[0], X_train.shape[0] * (1 - val_ratio), X_train.shape[0] * (val_ratio), \
           X_test.shape[0]


def save_model_performance_results(training_time, acc, loss, trained_count, validation_count, test_count, model_name):
    """Save and collect model performance results to a csv file."""

    df_training = pd.DataFrame(data=[
        [model_name, training_time, acc, loss, trained_count, validation_count, test_count]],
        columns=['model', 'run_time', 'test_accuracy', 'test_loss',
                 'trained_count', 'validation_count', 'test_count'])

    df_training.to_csv(f'outputs//digit_recognition//performance_results.csv', mode='a', header=False, index=False)


if __name__ == "__main__":
    model_name = 'cnn_architecture_8'
    img_size = 28

    save_folder_path = f'outputs//digit_recognition//{model_name}'
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    plot_mnist_images(nrows=3, ncols=5, show=False)

    training_time, acc, loss, trained_count, validation_count, test_count = train_cnn_network(epochs=50,
                                                                                              batch_size=128,
                                                                                              val_ratio=0.2,
                                                                                              early_stop_epochs=5,
                                                                                              img_size=img_size,
                                                                                              learning_rate=0.001,
                                                                                              shuffle=True,
                                                                                              model_name=model_name)
    save_model_performance_results(training_time, acc, loss,
                                   trained_count, validation_count, test_count,
                                   model_name=model_name)
