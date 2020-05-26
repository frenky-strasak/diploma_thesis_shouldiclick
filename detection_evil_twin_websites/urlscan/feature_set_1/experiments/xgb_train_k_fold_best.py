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

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from load_raw_data import prepare_data

def get_model():
    model = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        min_child_weight=5,
        n_jobs=3,
    )
    return model


def ratio_of_pos_and_neg(y_train: np.ndarray):
    positive_samples_count = np.sum(y_train)
    negative_samples_count = y_train.shape[0] - np.sum(y_train)
    assert y_train.shape[0] == positive_samples_count + negative_samples_count
    return positive_samples_count, negative_samples_count



def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    # axes[0].grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")

    return plt


only_good_features = False
if len(sys.argv) == 2:
    only_good_features = True
    result_folder = 'xgb_result_train_k_fold_fswfi'
else:
    result_folder = 'xgb_result_train_k_fold'

print('only_good_features: {}'.format(only_good_features))




positive_data_path = 'data/train_positive_X.txt'
negative_data_path = 'data/train_negative_X.txt'

data: Tuple[np.ndarray, np.ndarray] = prepare_data(positive_data_path, negative_data_path, normalize=False, standardize=False)
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



os.mkdir(result_folder)

log_file_name = 'log.txt'
log_file = open(result_folder + '/' + log_file_name, 'a')

def __print(text: str):
    log_file.write(text + '\n')
    print(text)


__print('All data:{}'.format(X.shape))


""" Learning curve """
print('------ Learning Curve ----------')
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = get_model()
plot_learning_curve(estimator, '', X, y, ylim=(0.7, 1.01),
                    cv=cv, n_jobs=2)
figure = plt.gcf()  # get current figure
# figure.set_size_inches(16, 9)
figure.set_size_inches(10, 6)
plt.savefig(result_folder + '/xgb_learning_curve.png', bbox_inches='tight')
plt.show()





K = 10
kf = StratifiedKFold(n_splits=K)
model_names = ['xgboost']
result_dict = {}
mean_result_dict = {}
for model_i, model_name in enumerate(model_names):
    mean_result_dict[model_name] = {}

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

        trained_model = get_model()

        trained_model.fit(X_train, y_train)

        """ ROC """
        viz = plot_roc_curve(trained_model, X_test, y_test,
                             name='ROC fold {}'.format(k_iteration),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        """ """

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
            label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    mean_result_dict[model_name]['mean_auc'] = mean_auc
    mean_result_dict[model_name]['std_auc'] = std_auc

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




for module, temp_dict in result_dict.items():
    __print(module)
    # mean_result_dict[module] = {}
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

    plot_error_bar(module, mean_list, std_list, x_values, metric_list)


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