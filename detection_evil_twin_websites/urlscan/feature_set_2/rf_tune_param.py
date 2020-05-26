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


from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from scipy import interp



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
    return X, y


def load_data(negative_data_path, positive_data_path):
    negative_X = np.load(negative_data_path)
    negative_X = shuffle_X(negative_X)

    positive_X = np.load(positive_data_path)
    positive_X = shuffle_X(positive_X)

    negative_X, positive_X = resize(negative_X, positive_X)

    X, y =  create_X_y(negative_X, positive_X)
    return X, y


positive_data_path = 'splited_data_310/train_positive.npy'
negative_data_path = 'splited_data_310/train_negative.npy'

data: Tuple[np.ndarray, np.ndarray] = load_data(positive_data_path, negative_data_path)
X = data[0]
y = data[1]


# idx = np.random.RandomState(seed=11).permutation(X.shape[0])
# idx = np.random.permutation(X.shape[0])
# X, y = X[idx], y[idx]


# X = X[:100]
# y = y[:100]

# print(X[0])
# print(X[1])
# print(X[2])
# print(X[3])



result_folder = 'rf_tune_param'
os.mkdir(result_folder)

log_file_name = 'log.txt'
log_file = open(result_folder + '/' + log_file_name, 'a')

def __print(text: str):
    log_file.write(text + '\n')
    print(text)

__print('All data:{}'.format(X.shape))

K = 10
kf = StratifiedKFold(n_splits=K)
model_names = ['randomforest']

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

rf_ts = time.time()
__print('\n\nRANDOM FOREST TUNNING:')
best_test_acc = {}
best_accuracy_value = 0
iteration = 1
model_name = model_names[0]
for n_estimators in [100, 500]:
    for max_depth in [None, 50, 100]:
        for min_samples_leaf in [1, 10]:
            __print('{}'.format(iteration))
            """ k-fold cross validation """
            test_acc_list = []
            train_acc_list = []
            fold_index = 0
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                __print('   K:{}'.format(fold_index))
                fold_index += 1

                __print('   X_train_size:{}'.format(X_train.shape))
                __print('   X_test_size: {}'.format(X_test.shape))
                __print('   all_check: {}'.format(X_test.shape[0] + X_train.shape[0]))

                train_positive_samples, train_negative_samples = ratio_of_pos_and_neg(y_train)
                test_positive_samples, test_negative_samples = ratio_of_pos_and_neg(y_test)
                __print('   train_positive_samples: {}'.format(train_positive_samples))
                __print('   train_negative_samples: {}'.format(train_negative_samples))
                __print('   all_check_train: {}'.format(train_negative_samples + train_positive_samples))
                __print('   test_positive_samples: {}'.format(test_positive_samples))
                __print('   test_negative_samples: {}'.format(test_negative_samples))
                __print('   all_check_test: {}'.format(test_negative_samples + test_positive_samples))

                xg_boost = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features="sqrt",
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=3,
                )
                xg_boost.fit(X_train, y_train)

                y_pred_train = xg_boost.predict(X_train)
                y_pred_test = xg_boost.predict(X_test)

                accuracy_train = accuracy_score(y_train, y_pred_train)
                accuracy_test = accuracy_score(y_test, y_pred_test)

                test_acc_list.append(accuracy_test)
                train_acc_list.append(accuracy_train)

            """ END k-fold cross validation """
            test_ac_mean = np.array(test_acc_list).mean()
            train_acc_mean = np.array(train_acc_list).mean()

            __print('RF: PARAM: n_estimators:{}, max_depth:{}, min_samples_leaf: {}'.format(n_estimators, max_depth, min_samples_leaf))
            __print('RF: Result: train_acc: {}, test_acc: {}'.format(train_acc_mean, test_ac_mean))

            if best_accuracy_value < test_ac_mean:
                best_test_acc = {
                    'iteration': iteration,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'train_acc': train_acc_mean,
                    'test_acc': test_ac_mean
                }
                best_accuracy_value = test_ac_mean

            iteration += 1

__print('########### BEST RESULT #############')
for key, value in best_test_acc.items():
    __print('{} {}'.format(key, value))


final_time: float = time.time() - rf_ts
__print('time {}'.format(final_time / 3600.0))



