import itertools
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATH = '/unibuc-2019-m2-cv'


def form_img_path(x, extension='', path=''):
    prefix = '0' * (6-len(str(x)))
    if path:
        path = path + "/"
    return f'{path}{prefix}{x}{extension}'


def load_dataset(train_labels_path=f'{PATH}/train_labels.txt', img_path=f'{PATH}/data/data'):
    df = pd.read_csv(train_labels_path)
    df['img'] = df.apply(lambda row: cv2.imread(
        form_img_path(row['id'], extension='.png', path=img_path)), axis=1)

    X = df['img'].values
    X = np.concatenate(X).reshape(-1, 224, 224, 3)

    y = df['class'].values

    return X, y


def output_bias(y):
    neg, pos = np.bincount(y)
    return np.log([pos/neg])


def class_weight(y):
    neg, pos = np.bincount(y)
    total = neg + pos
    weight_for_0 = (1.0/neg)*(total)/2.0
    weight_for_1 = (1.0/pos)*(total)/2.0

    return {0: weight_for_0, 1: weight_for_1}


def parse_predict(y_pred):
    return (y_pred.flatten() > 0.5).astype(int)


# predict on test data
def run_test_data(model,  output_file, img_path=f'{PATH}/data/data'):
    img_list = [form_img_path(x, extension='.png', path=img_path)
                for x in range(17001, 22150)]
    X_test = np.array([cv2.imread(img) for img in img_list])

    y = model.predict(X_test, batch_size=32)

    ans_df = pd.DataFrame({'id': [form_img_path(x) for x in range(17001, 22150)],
                           'class': parse_predict(y)})
    ans_df.to_csv(output_file, index=False)


# combine the result of the folds by selecting the class with most votes
def evaluate_folds(model_name, path):
    folds = []
    for fold in range(3):
        folds.append(pd.read_csv(
            f'{path}/{fold}.csv'))
    df = pd.concat(folds)
    result_df = df.groupby('id').agg(lambda x: x.value_counts().index[0])
    result_df.to_csv(f'{path}/{model_name}.csv')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          savefig=None):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    plt.clf()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if savefig:
        plt.savefig(savefig)

    return plt


def plot_metrics(history, savefig=None):
    plt.clf()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace('_', ' ').capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch,
                 history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle='--', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()

    if savefig:
        plt.savefig(savefig)

    return plt
