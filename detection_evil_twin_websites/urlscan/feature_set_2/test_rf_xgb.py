import sys
import os
import argparse
import time

from  numpy import loadtxt
from numpy import sort
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix

from typing import Tuple
from datetime import datetime


def get_model(i: int):
    models = [
        XGBClassifier(
            n_estimators=500,
            max_depth=50,
            min_child_weight=10,
            n_jobs=2,
        ),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=100,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=2,
        )
    ]
    return models[i]


def ratio_of_pos_and_neg(y_train: np.ndarray):
    positive_samples_count = np.sum(y_train)
    negative_samples_count = y_train.shape[0] - np.sum(y_train)
    assert y_train.shape[0] == positive_samples_count + negative_samples_count
    return positive_samples_count, negative_samples_count


def shuffle_X(X):
    idx = np.random.RandomState(seed=11).permutation(X.shape[0])
    # idx = np.random.permutation(X.shape[0])
    X = X[idx]
    return X


def resize(negative_X, positive_X):
    min_size = int(min(negative_X.shape[0], positive_X.shape[0]))
    negative_X = negative_X[:min_size]
    positive_X = positive_X[:min_size]
    assert positive_X.shape[0] == negative_X.shape[0]
    return negative_X, positive_X


def create_X_y(negative_X, positive_X):
    print(positive_X.shape[1])
    print(negative_X.shape[1])
    assert positive_X.shape[1] == 310, 'Diif size of columns!'
    assert negative_X.shape[1] == 310, 'Diif size of columns!'
    assert positive_X.shape[0] == negative_X.shape[0]
    y_positive = np.ones(positive_X.shape[0])
    y_negative = np.zeros(negative_X.shape[0])
    X = np.concatenate((positive_X, negative_X))
    assert X.shape[0] == negative_X.shape[0] + positive_X.shape[0]
    y = np.concatenate((y_positive, y_negative))
    assert y.shape[0] == y_negative.shape[0] + y_positive.shape[0]
    return X,y


def load_data(negative_data_path, positive_data_path):
    negative_X = np.load(negative_data_path)
    negative_X = shuffle_X(negative_X)

    positive_X = np.load(positive_data_path)
    positive_X = shuffle_X(positive_X)

    negative_X, positive_X = resize(negative_X, positive_X)

    X, y =  create_X_y(negative_X, positive_X)
    return X, y


train_positive_data_path = 'splited_data_310/train_positive.npy'
train_negative_data_path = 'splited_data_310/train_negative.npy'
data: Tuple[np.ndarray, np.ndarray] = load_data(train_positive_data_path, train_negative_data_path)
X_train = data[0]
y_train = data[1]



test_positive_data_path = 'splited_data_310/test_positive.npy'
test_negative_data_path = 'splited_data_310/test_negative.npy'
data: Tuple[np.ndarray, np.ndarray] = load_data(test_positive_data_path, test_negative_data_path)
X_test = data[0]
y_test = data[1]


model_names = ['xbboost', 'randomforest']
for model_i, model_name in enumerate(model_names):
    trained_model = get_model(model_i)
    trained_model.fit(X_train, y_train)

    y_pred_train = trained_model.predict(X_train)
    y_pred_test = trained_model.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_test).ravel()
