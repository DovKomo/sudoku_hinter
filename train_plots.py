import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from scikitplot.metrics import plot_roc


def plot_model_performance(history, final_test_acc, final_test_loss, model_name='cnn_architecture_2'):
    """Plots the model performance (accuracy and loss) in each epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, final in zip([ax1, ax2], ['accuracy', 'loss'], [final_test_acc, final_test_loss]):
        ax.plot(history.history[f'{metric}'])
        ax.plot(history.history[f'val_{metric}'])
        ax.set_title(f'model {metric}: {round(final * 100, 2)}%')
        ax.set(xlabel='epoch', ylabel=f'{metric}')
    plt.legend(['train', 'val'], loc='upper left')

    plt.savefig(f'outputs//digit_recognition//{model_name}//model_performance.png')
    plt.clf()
    plt.close(fig)


def plot_data_distribution(y_train, y_test, model_name='cnn_architecture_2'):
    """Plots data distribution for each category."""

    classes = list(np.unique(y_train))
    y_train_values = [list(y_train).count(class_i) for class_i in classes]
    y_test_values = [list(y_test).count(class_i) for class_i in classes]

    # plotting
    x = np.arange(len(classes))  # the label locations
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, y_train_values, width=0.3, label='Train', color='forestgreen', edgecolor='white')
    ax.bar(x + 0.1, y_test_values, width=0.3, label='Test', color='teal', edgecolor='white')

    for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Counts")
    ax.set_xlabel('Digits')
    ax.set_title('Distribution of train/test datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'outputs//digit_recognition//{model_name}//train_test_data_distribution.png', bbox_inches='tight')
    plt.clf()


def get_confusion_matrix(y_pred, y_test, classes, model_name):
    """Calculates and plots confusion matrix."""

    sns.set(font_scale=1.5)

    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=3)
    con_mat_df = pd.DataFrame(con_mat_norm, index=list(range(len(classes))),
                              columns=list(range(len(classes))))

    plt.figure(figsize=(12, 12))
    plt.tight_layout()
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, linewidth=1, xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label', size=16)
    plt.xlabel('Predicted label', size=16)
    plt.suptitle('Normalized confusion matrix', y=0.94, size=28)

    plt.savefig(f'outputs//digit_recognition//{model_name}//confusion_matrix.png', bbox_inches='tight')
    plt.clf()


def get_roc_curves(y_pred_raw, y_test, model_name):
    """Plots and saves ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(16, 12))
    plot_roc(y_test, y_pred_raw, ax=ax)
    plt.savefig(f'outputs//digit_recognition//{model_name}//roc_curve.png', bbox_inches='tight')
    plt.close(fig)
    plt.clf()
