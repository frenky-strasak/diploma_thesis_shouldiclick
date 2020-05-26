import os
from typing import Tuple
import numpy as np
import random
import argparse
import time

import umap
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
import seaborn as sns

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D


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


positive_data_path = 'splited_data_310/positive.npy'
negative_data_path = 'splited_data_310/negative.npy'

data: Tuple[np.ndarray, np.ndarray] = load_data(positive_data_path, negative_data_path)
X = data[0]
y = data[1]

print(X.shape)
# X = X[:100]
# y = y[:100]



def umap_2(X: np.ndarray, y: np.ndarray, title: str, to_show: bool = True, to_save: bool = False,  metric='euclidean',
           n_neighbors=15, min_dist=0.1):
    print('UMAP 2')
    names = ['evil twin', 'legitimate']
    colors = ['red', 'green']

    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
    reducer = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist)
    embedding = reducer.fit_transform(X)
    target_ids = range(len(names))
    for i, c, label in zip(target_ids, colors, names):
        plt.scatter(embedding[y == i, 0], embedding[y == i, 1], c=c, label=label, alpha=0.1)

    # plt.gca().set_aspect('equal', 'datalim')
    plt.title('{}'.format(title), fontsize=24)

    def update(handle, orig):
        handle.update_from(orig)
        handle.set_alpha(1)

    plt.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update),
                            plt.Line2D: HandlerLine2D(update_func=update)})
    # plt.legend(loc="best", shadow=False, scatterpoints=1)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)


    if to_save:
        plt.savefig('plot/' + 'evil_twin_umap_set_2_2' + '.png', bbox_inches='tight')
    if to_show:
        plt.show()



umap_2(X, y, 'UMAP - Evil twin and Legitimate websites', to_save=True)
