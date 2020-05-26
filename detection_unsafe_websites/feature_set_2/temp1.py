from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split

X = []
y = []
for i in range(20):
    X.append((i,i))
    if i < 10:
        y.append(0)
    else:
        y.append(1)

# print(X)
# print(y)

X = np.array(X)
y = np.array(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(y_train)
print(np.sum(y_train))
print(y_test)
print(np.sum(y_test))
#
#
#
# K = 2
# kf = StratifiedKFold(n_splits=K)
#
# for train_index, test_index in kf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     print(X_train)
#     # print(X_test)


# with open('/home/frenky/Documents/Skola/Magistr/diplomka/shouldiclick_thesis/2018_process/new/all/FSWFI/randomforest_4/best_features.txt') as f:
#     for line in f:
#         line = line.rstrip()
#         if line == '':
#             continue
#         splited = line.split(' ')
#         if len(splited) == 2:
#             name = line.split(' ')[0]
#         else:
#             name = splited[0] + ' ' + splited[1]
#         print(name)


# def read_feature_names():
#     features_names_and_indexes = []
#     with open('feature_names_310.txt') as f:
#         for i, line in enumerate(f):
#             line = line.rstrip()
#             if line == '':
#                 continue
#             features_names_and_indexes.append((line, i))
#     return features_names_and_indexes
#
#
# feature_names = read_feature_names()
#
# for f in feature_names:
#     print(f)


