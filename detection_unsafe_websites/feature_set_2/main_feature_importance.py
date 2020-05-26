"""
READ IT !!!
source: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
"""




import sys
import os
import argparse
import time

from numpy import loadtxt
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

from typing import Tuple
from datetime import datetime



class IoWriter:

    def __init__(self, sub_folder_name: str):
        os.mkdir(sub_folder_name)
        self.log_file = open(sub_folder_name + '/' + 'log.txt', 'a')
        self.best_feature_file = open(sub_folder_name + '/' + 'best_features.txt', 'w')
        self.best_feature_file_only_names = open(sub_folder_name + '/' + 'best_features_only_names.txt', 'w')
        self.all_feature_file = open(sub_folder_name + '/' + 'all_features.txt', 'w')
        self.best_feature_freq = open(sub_folder_name + '/' + 'best_feature_freq.txt', 'w')
        self.ts_start = time.time()

    def write_log(self, text: str):
        self.log_file.write(text + '\n')
        print(text)

    def write_stats(self, accuracy_train, accuracy_test, best_accuracy, best_threshold, best_features_names: list, all_features_names: list, features_frequency: dict):
        self.write_log('Accuracy from all train data: {}'.format(accuracy_train))
        self.write_log('Accuracy from all test data: {}'.format(accuracy_test))
        self.write_log('best accuracy: {} was gained with: {} threshold'.format(best_accuracy, best_threshold))
        for importance_value, feature_name in sorted(best_features_names):
            self.best_feature_file.write('{} {}\n'.format(feature_name, importance_value))
            self.best_feature_file_only_names.write('{}\n'.format(feature_name))
        for importance_value, feature_name in sorted(all_features_names):
            self.all_feature_file.write('{} {}\n'.format(feature_name, importance_value))
        for best_feature, freq in features_frequency.items():
            self.best_feature_freq.write('{} {}\n'.format(best_feature, freq))
        self.write_log('Duration: {}s'.format(time.time() - self.ts_start))


def get_model(i: int):
    models = [
        XGBClassifier(
            n_estimators=500,
            max_depth=10,
            min_child_weight=10,
            n_jobs=3,
        ),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=100,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=3,
        ),
    ]
    return models[i]

# def get_model(i: int):
#     models = [
#         XGBClassifier(n_jobs=2,),
#         RandomForestClassifier(n_jobs=2,)
#     ]
#     return models[i]


def select_features_by_thresh(trained_model, model_int: int, best_accuracy: float, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    best_threshold = 0
    best_feature_selection = np.ones(X_train.shape[1], dtype=bool)
    thresholds = sort(trained_model.feature_importances_)
    for thresh in thresholds:
        if thresh == 0:
            continue
        selection: SelectFromModel = SelectFromModel(trained_model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        new_model = get_model(model_int)
        new_model.fit(select_X_train, y_train)

        select_X_test = selection.transform(X_test)
        y_pred = new_model.predict(select_X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # __print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
            # get_support() -> [ True False False False False]
            best_feature_selection = selection.get_support()

    return best_accuracy, best_threshold, best_feature_selection


def train(model_int: int, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, features_and_indexes: list):
    trained_model = get_model(model_int)

    trained_model.fit(X_train, y_train)

    y_pred_train = trained_model.predict(X_train)
    y_pred_test = trained_model.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    best_accuracy, best_threshold, best_feature_selection = select_features_by_thresh(trained_model, model_int, accuracy_test, X_train, X_test, y_train, y_test)

    index_2 = 0
    best_features_names = []
    for importance_value, (feature_name, index) in zip(trained_model.feature_importances_, features_and_indexes):
        assert index_2 == index, 'index != index_2: {} {}'.format(index, index_2)
        alived_feature_1 = bool(best_feature_selection[index_2])
        alived_feature_2 = bool(float(importance_value) >= float(best_threshold))
        assert alived_feature_1 == alived_feature_2, 'alived_feature_1 != alived_feature_2: {}: {}'.format(alived_feature_1, alived_feature_2)
        # print('-------')
        # print(alived_feature_1)
        # print(type(alived_feature_1))
        # print(alived_feature_2)
        # print(type(alived_feature_2))
        # print(alived_feature_1 is True)
        # print(alived_feature_2 is True)
        # assert alived_feature_1 is True and alived_feature_2 is True, 'alived_feature_1 != alived_feature_2: {}: {}'.format(alived_feature_1, alived_feature_2)
        if alived_feature_1 is True and alived_feature_2 is True:
            best_features_names.append((importance_value, feature_name))
        index_2 += 1

    return accuracy_train, accuracy_test, best_accuracy, best_threshold, best_features_names, trained_model.feature_importances_


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

def read_feature_names():
    features_names_and_indexes = []
    with open('feature_names_310.txt') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if line == '':
                continue
            features_names_and_indexes.append((line, i))
    return features_names_and_indexes

"""
Beginning of the code
"""
ts_start = time.time()

now = datetime.now()
# folder_name = now.strftime("%Y_%m_%d_%H_%M_%S")
folder_name = now.strftime("FSWFI")
os.mkdir(folder_name)


positive_data_path = 'splited_data_310/train_positive.npy'
negative_data_path = 'splited_data_310/train_negative.npy'

features_and_indexes = read_feature_names()

data: Tuple[np.ndarray, np.ndarray] = load_data(positive_data_path, negative_data_path)
X = data[0]
y = data[1]


""" shuffle data"""
# idx = np.random.permutation(X.shape[0])
# X, y = X[idx], y[idx]


K = 10
# K = [3]
# models = [XGBClassifier(), RandomForestClassifier()]
model_names = ['xbboost', 'randomforest']
# model_names = ['xbboost']
for i in range(0, len(model_names)):
    kf = StratifiedKFold(n_splits=K)
    result_list = []
    features_importance_numpy_arr = np.zeros(X.shape[1])

    sub_folder_name = model_names[i] + '_' + str(K)
    io_writer = IoWriter(folder_name + '/' + sub_folder_name)
    io_writer.write_log('####################')
    io_writer.write_log('model: {},  K={}'.format(model_names[i], K))
    io_writer.write_log('All data: {}'.format(X.shape[0]))
    pos, neg = ratio_of_pos_and_neg(y)
    io_writer.write_log('Positive:{}, Negative:{}'.format(pos, neg))
    features_frequency = {}
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_positive_samples, train_negative_samples = ratio_of_pos_and_neg(y_train)
        test_positive_samples, test_negative_samples = ratio_of_pos_and_neg(y_test)
        io_writer.write_log('train data ratio: positive: {}, negative: {}'.format(train_positive_samples,
                                                                                  train_negative_samples))
        io_writer.write_log(
            'test data ratio: positive: {}, negative: {}'.format(test_positive_samples, test_negative_samples))

        accuracy_train, accuracy_test, best_accuracy, best_threshold, best_features_names,\
        feature_importances_ = train(i, X_train, X_test, y_train,y_test, features_and_indexes)

        result_list.append((accuracy_train, accuracy_test, best_accuracy, best_threshold))
        features_importance_numpy_arr += np.array(feature_importances_)

        for importance_value, feature_name in best_features_names:
            if features_frequency.get(feature_name, None) is None:
                features_frequency[feature_name] = 1
            else:
                features_frequency[feature_name] += 1


    features_importance_numpy_arr = features_importance_numpy_arr / K

    np_result_list = np.array(result_list)
    accuracy_train = np_result_list[:, 0].mean()
    accuracy_test = np_result_list[:, 1].mean()
    best_accuracy = np_result_list[:, 2].mean()
    best_threshold = np_result_list[:, 3].mean()

    """ Find the best feature from means of K folds """
    best_features_names = []
    all_features_names = []
    for importance_value, (feature_name, index) in zip(list(features_importance_numpy_arr),
                                                       features_and_indexes):
        if importance_value >= best_threshold:
            best_features_names.append((importance_value, feature_name))
        all_features_names.append((importance_value, feature_name))

    io_writer.write_stats(accuracy_train, accuracy_test, best_accuracy, best_threshold, best_features_names, all_features_names, features_frequency)


print('Process length: {}'.format((time.time() - ts_start) / 3600.0))




# model = RandomForestClassifier()
#
# x_pos = 1 * np.random.random_sample((100, 5))
# x_neg = 5 * np.random.random_sample((100, 5))
#
#
# y_pos = np.ones(100)
# y_neg = np.zeros(100)
#
# X = np.concatenate((x_pos, x_neg))
# y = np.concatenate((y_pos, y_neg))
#
#
#
#
# # split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
#
# model.fit(X_train, y_train)
#
# pred = model.predict(X_test)
# print(pred)
#
# print(model.feature_importances_)
#
# thresh = model.feature_importances_[0]
#
# selection = SelectFromModel(model, threshold=thresh, prefit=True)
#
# supp = selection.get_support()
# print(supp)
#
# print(X_train[0])
# select_X_train = selection.transform(X_train)
# print(select_X_train[0])
