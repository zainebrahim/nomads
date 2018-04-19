import itertools

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale

import sys
sys.path.append('/Users/jaewonc78/git/lol/lol')
from lol import LOL


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(
        tick_marks,
        classes,
        rotation=45,
    )
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_cm(cm):
    arr = np.array(cm)
    arr = arr.astype('float') / arr.sum(axis=(2), keepdims=True)
    arr_mean = arr.mean(axis=0)
    arr_std = arr.std(axis=0)

    return arr_mean, arr_std


def rf(X, y, normalize=True, max_features='auto', n_splits=5, n_repeats=5):

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    cm = []

    for train_idx, test_idx in kfold.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        if normalize:
            X_train, X_test = scale(X[train_idx]), scale(X[test_idx])
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        rf = RandomForestClassifier(
            class_weight='balanced', max_features=max_features)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        cm.append(confusion_matrix(y_test, pred))

    cm, _ = compute_cm(cm)
    return cm


def lda(X, y, normalize=True, n_splits=5, n_repeats=5):
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    cm = []

    for train_idx, test_idx in kfold.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        if normalize:
            X_train, X_test = scale(X[train_idx]), scale(X[test_idx])
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        l = LinearDiscriminantAnalysis()
        l.fit(X_train, y_train)
        pred = l.predict(X_test)
        cm.append(confusion_matrix(y_test, pred))

    cm, _ = compute_cm(cm)
    return cm


def qda(X, y, normalize=True, n_splits=5, n_repeats=5):
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    cm = []

    for train_idx, test_idx in kfold.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        if normalize:
            X_train, X_test = scale(X[train_idx]), scale(X[test_idx])
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        l = QuadraticDiscriminantAnalysis()
        l.fit(X_train, y_train)
        pred = l.predict(X_test)
        cm.append(confusion_matrix(y_test, pred))

    cm, _ = compute_cm(cm)
    return cm


def lda_lol(X, y, normalize=True, n_components=None, n_splits=5, n_repeats=5):
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    cm = []

    for train_idx, test_idx in kfold.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        if normalize:
            X_train, X_test = scale(X[train_idx]), scale(X[test_idx])
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        p = LOL(n_components=n_components)

        X_train = p.fit_transform(X_train, y_train)
        X_test = p.transform(X_test)

        l = LinearDiscriminantAnalysis()
        l.fit(X_train, y_train)
        pred = l.predict(X_test)
        cm.append(confusion_matrix(y_test, pred))

    cm, _ = compute_cm(cm)
    return cm


def qda_lol(X, y, normalize=True, n_components=None, n_splits=5, n_repeats=5):
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    cm = []

    for train_idx, test_idx in kfold.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        if normalize:
            X_train, X_test = scale(X[train_idx]), scale(X[test_idx])
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        p = LOL(n_components=n_components)
        X_train = p.fit_transform(X_train, y_train)
        X_test = p.transform(X_test)

        #features = p.explained_variance_ratio_ < 0.9

        #X_train = X_train[:, features]
        #X_test = X_test[:, features]

        l = QuadraticDiscriminantAnalysis()
        l.fit(X_train, y_train)
        pred = l.predict(X_test)
        cm.append(confusion_matrix(y_test, pred))

    cm, _ = compute_cm(cm)
    return cm