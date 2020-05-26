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

def read_feature_names():
    features_names_and_indexes = []
    with open('feature_names_310.txt') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if line == '':
                raise NameError('empty line in feature_names_310.txt')
            # features_names_and_indexes.append((line, i))
            features_names_and_indexes.append(line)
    return features_names_and_indexes


def read_useful_feature_names():
    features_names_and_indexes = []
    with open('alives_features_for_rf.txt') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if line == '':
                raise NameError('Empty line in alives_features.txt!')
            # features_names_and_indexes.append((line, i))
            features_names_and_indexes.append(line)
    return features_names_and_indexes


def remove_features(X: np.ndarray):
    """ Remove features without importance """
    print('Removing features...')
    all_feature_names = read_feature_names()
    alives_features = read_useful_feature_names()
    print('all_features: {}'.format(len(all_feature_names)))
    print('good_features: {}'.format(len(alives_features)))
    """ Validity check """
    for alives_feature in alives_features:
        assert alives_feature in all_feature_names, 'Invalid name for alive feature: {}'.format(alives_feature)
    new_X = np.zeros((X.shape[0], len(alives_features)))
    new_index = 0
    for index, feature, in enumerate(all_feature_names):
        if feature in alives_features:
            new_X[:, new_index] = X[:, index]
            new_index += 1

    assert new_X.shape[1] == len(alives_features)
    return new_X


def load_data(negative_data_path, positive_data_path, only_good_features: bool = False):
    negative_X = np.load(negative_data_path)
    negative_X = shuffle_X(negative_X)

    positive_X = np.load(positive_data_path)
    positive_X = shuffle_X(positive_X)

    negative_X, positive_X = resize(negative_X, positive_X)

    X, y = create_X_y(negative_X, positive_X)

    if only_good_features is True:
        X = remove_features(X)

    return X, y


def get_feature_names_2():
    feature_names = []
    all_feature_names = read_feature_names()
    alives_features = read_useful_feature_names()
    index_2 = 0
    for index, feature, in enumerate(all_feature_names):
        if feature in alives_features:
            feature_names.append((feature, index_2))
            index_2 += 1
    return feature_names


result_folder = 'test_result_rf'

os.mkdir(result_folder)

log_file_name = 'log.txt'
log_file = open(result_folder + '/' + log_file_name, 'a')

def __print(text: str):
    log_file.write(text + '\n')
    print(text)



features_and_indexes = get_feature_names_2()

train_positive_data_path = 'splited_data_310/train_positive.npy'
train_negative_data_path = 'splited_data_310/train_negative.npy'
data: Tuple[np.ndarray, np.ndarray] = load_data(train_positive_data_path, train_negative_data_path, True)
X_train = data[0]
y_train = data[1]
print('training data: {}'.format(X_train.shape))


test_positive_data_path = 'splited_data_310/test_positive.npy'
test_negative_data_path = 'splited_data_310/test_negative.npy'
data: Tuple[np.ndarray, np.ndarray] = load_data(test_positive_data_path, test_negative_data_path, True)
X_test = data[0]
y_test = data[1]
print('testing data: {}'.format(X_test.shape))


assert len(features_and_indexes) == X_train.shape[1]
assert len(features_and_indexes) == X_test.shape[1]


model_name = 'random_forest'
result_dict = {}
result_dict[model_name] = {
    'train_acc': [],
    'test_acc': [],
    'TN': [],
    'TP': [],
    'FP': [],
    'FN': [],
    'FPR': [],
    'FDR': [],
    'FNR': [],
    'TPR': [],
    'SPC': [],
    'PPV': [],
    'NPV': [],
}

__print(model_name + 'X_train_size:{}'.format(X_train.shape))
__print(model_name + 'X_test_size: {}'.format(X_test.shape))
__print(model_name + 'all_check: {}'.format(X_test.shape[0] + X_train.shape[0]))

train_positive_samples, train_negative_samples = ratio_of_pos_and_neg(y_train)
test_positive_samples, test_negative_samples = ratio_of_pos_and_neg(y_test)

__print(model_name + 'train_positive_samples: {}'.format(train_positive_samples))
__print(model_name + 'train_negative_samples: {}'.format(train_negative_samples))
__print(model_name + 'all_check_train: {}'.format(train_negative_samples + train_positive_samples))
__print(model_name + 'test_positive_samples: {}'.format(test_positive_samples))
__print(model_name + 'test_negative_samples: {}'.format(test_negative_samples))
__print(model_name + 'all_check_test: {}'.format(test_negative_samples + test_positive_samples))



trained_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=100,
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=3
            )


trained_model.fit(X_train, y_train)

feature_importance_list = trained_model.feature_importances_

y_pred_train = trained_model.predict(X_train)
y_pred_test = trained_model.predict(X_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
TN, FP, FN, TP = confusion_matrix(y_test, y_pred_test).ravel()

__print(model_name + 'train_acc: {}'.format(accuracy_train))
__print(model_name + 'test_acc: {}'.format(accuracy_test))
__print(model_name + 'TN:{}'.format(TN))
__print(model_name + 'FP:{}'.format(FP))
__print(model_name + 'FN:{}'.format(FN))
__print(model_name + 'TP:{}'.format(TP))

result_dict[model_name]['train_acc'].append(accuracy_train)
result_dict[model_name]['test_acc'].append(accuracy_test)
result_dict[model_name]['TN'].append(TN)
result_dict[model_name]['TP'].append(TP)
result_dict[model_name]['FP'].append(FP)
result_dict[model_name]['FN'].append(FN)

_test_acc = (TP + TN) / (FP + TN + FN + TP)
__print('_test_acc: {}'.format(_test_acc))
# FPR = FP / (FP + TN)
FPR = FP / (FP + TN)
__print('False Positive Rate FPR: {}'.format(FPR))
# FDR = FP / (FP + TP)
FDR = FP / (FP + TP)
__print('False Discovery Rate FDR: {}'.format(FDR))
# FNR = FN / (FN + TP)
FNR = FN / (FN + TP)
__print('False Negative Rate FNR: {}'.format(FNR))
# TPR = TP / (TP + FN)
TPR = TP / (TP + FN)
__print('Sensitivity TPR: {}'.format(TPR))
# SPC = TN / (FP + TN)
SPC = TN / (FP + TN)
__print('Specificity SPC: {}'.format(SPC))
# PPV = TP / (TP + FP)
PPV = TP / (TP + FP)
__print('Precision PPV: {}'.format(PPV))
# NPV = TN / (TN + FN)
NPV = TN / (TN + FN)
__print('Negative Predictive Value NPV: {}'.format(NPV))

result_dict[model_name]['FPR'].append(FPR)
result_dict[model_name]['FDR'].append(FDR)
result_dict[model_name]['FNR'].append(FNR)
result_dict[model_name]['TPR'].append(TPR)
result_dict[model_name]['SPC'].append(SPC)
result_dict[model_name]['PPV'].append(PPV)
result_dict[model_name]['NPV'].append(NPV)



for model_name, temp_dict in result_dict.items():
    for metric, value_list in temp_dict.items():
        print('{}: {}'.format(metric, value_list[0]))

""" Feature importance """
assert len(features_and_indexes) == len(list(feature_importance_list))
feature_importances = []
for importance_value, (feature_name, index) in zip(list(feature_importance_list), features_and_indexes):
    feature_importances.append((importance_value, feature_name))

with open(result_folder + '/feature_importance.txt', 'w') as f:
    for importance_value, feature_name in sorted(feature_importances):
        f.write('{}: {}\n'.format(feature_name, importance_value))