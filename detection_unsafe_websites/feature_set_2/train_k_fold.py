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
import matplotlib
from scipy import interp




def get_model(i: int):
    models = [
        XGBClassifier(),
        RandomForestClassifier()
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


positive_data_path = 'splited_data_310/train_positive.npy'
negative_data_path = 'splited_data_310/train_negative.npy'

data: Tuple[np.ndarray, np.ndarray] = load_data(positive_data_path, negative_data_path)
X = data[0]
y = data[1]


# X = X[:100]
# y = y[:100]


idx = np.random.RandomState(seed=11).permutation(X.shape[0])
# idx = np.random.permutation(X.shape[0])
X, y = X[idx], y[idx]




# print(X[0])
# print(X[1])
# print(X[2])
# print(X[3])


result_folder = 'result_train_k_fold'
os.mkdir(result_folder)

log_file_name = 'log.txt'
log_file = open(result_folder + '/' + log_file_name, 'a')

def __print(text: str):
    log_file.write(text + '\n')
    print(text)


__print('All data:{}'.format(X.shape))

K = 10

kf = StratifiedKFold(n_splits=K)

model_names = ['xbboost', 'randomforest']
# model_names = ['randomforest']

result_dict = {}
for model_i, model_name in enumerate(model_names):
    __print('\n\nmodel name:{}'.format(model_name))
    k_iteration = 1
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

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for train_index, test_index in kf.split(X, y):
        __print(model_name + '\nK:{}'.format(k_iteration))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
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

        trained_model = get_model(model_i)

        trained_model.fit(X_train, y_train)

        """ ROC """
        viz = plot_roc_curve(trained_model, X_test, y_test,
                             name='ROC fold {}'.format(k_iteration),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

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

        k_iteration += 1

    """ plot AUC """

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC with cross validation")
    ax.legend(loc="lower right")

    figure = plt.gcf()  # get current figure
    # figure.set_size_inches(32, 18)
    # figure.set_size_inches(16, 9)
    figure.set_size_inches(10, 6)
    # plt.savefig(result_folder + '/auc_roc_{}.png'.format(model_name))
    plt.savefig(result_folder + '/tight_auc_roc_{}.png'.format(model_name), bbox_inches='tight')
    plt.show()



mean_result_dict = {}
for module, temp_dict in result_dict.items():
    __print(module)
    mean_result_dict[module] = {}
    for metric, value_list in temp_dict.items():
        assert len(value_list) == K
        res_mean = np.array(value_list).mean()
        res_std = np.array(value_list).std()
        # __print('{} {}'.format(metric, res))
        mean_result_dict[module][metric] = res_mean
        mean_result_dict[module][metric + '_std'] = res_std




def plot_error_bar(model_name, means, errors, x, x_names):
    plt.figure()
    plt.errorbar(x, means, yerr=errors, xlolims=True, fmt='--o')
    plt.xticks(x, x_names)
    plt.title("Mean and STD of Accuracy on testing data with K-Fold cross validation")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)
    plt.savefig(result_folder + '/test_acc_mean_std_{}.png'.format(model_name), bbox_inches='tight')


# Plot error bar
metric_list = ['train_acc', 'test_acc', 'FPR', 'FNR']
x_values = [x for x in range(len(metric_list))]
for module, temp_dict in mean_result_dict.items():
    mean_list = []
    std_list = []
    for metric in metric_list:
        mean = temp_dict[metric]
        std = temp_dict[metric + '_std']
        mean_list.append(mean)
        std_list.append(std)

    # plot_error_bar(module, mean_list, std_list, x_values, metric_list)


__print('\n\n\nFINAL RESULT:')
for model_name in model_names:
    __print(model_name)

    FP = mean_result_dict[model_name]['FP']
    TP = mean_result_dict[model_name]['TP']
    FN = mean_result_dict[model_name]['FN']
    TN = mean_result_dict[model_name]['TN']

    for metric, value_mean in mean_result_dict[model_name].items():
        __print('{}: {}'.format(metric, value_mean))

    # _test_acc = (TP + TN) / (FP + TN + FN + TP)
    # __print('_test_acc: {}'.format(_test_acc))
    #
    # # FPR = FP / (FP + TN)
    # FPR = FP / (FP + TN)
    # __print('False Positive Rate FPR: {}'.format(FPR))
    #
    # # FDR = FP / (FP + TP)
    # FDR = FP / (FP + TP)
    # __print('False Discovery Rate FDR: {}'.format(FDR))
    #
    # # FNR = FN / (FN + TP)
    # FNR = FN / (FN + TP)
    # __print('False Negative Rate FNR: {}'.format(FNR))
    #
    # # TPR = TP / (TP + FN)
    # TPR = TP / (TP + FN)
    # __print('Sensitivity TPR: {}'.format(TPR))
    #
    # # SPC = TN / (FP + TN)
    # SPC = TN / (FP + TN)
    # __print('Specificity SPC: {}'.format(SPC))
    #
    # # PPV = TP / (TP + FP)
    # PPV = TP / (TP + FP)
    # __print('Precision PPV: {}'.format(PPV))
    #
    # # NPV = TN / (TN + FN)
    # NPV = TN / (TN + FN)
    # __print('Negative Predictive Value NPV: {}'.format(NPV))