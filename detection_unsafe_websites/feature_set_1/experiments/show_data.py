from load_raw_data import prepare_data
from load_raw_data import get_feature_names_and_indexes

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

# from numba import set_num_threads
import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 2

# set_num_threads(2)

# positive_data_path = '/home/frenky/Documents/Skola/Magistr/diplomka/shouldiclick_thesis/process/urlscan/xgboost_experiments/create_matrix/positive_X.txt'
# negative_data_path = '/home/frenky/Documents/Skola/Magistr/diplomka/shouldiclick_thesis/process/urlscan/xgboost_experiments/create_matrix/negative_X.txt'
#
#
# data: Tuple[np.ndarray, np.ndarray] = prepare_data(positive_data_path, negative_data_path, normalize=False)
#
# X = data[0]
# y = data[1]

NUMBA_NUM_THREADS = 2


# def generate_data(a, b, n):
#     l = []
#     for _ in range(n):
#         sample = []
#         for _ in range(10):
#             x = random.randint(a, b)
#             sample.append(float(x))
#         l.append(sample)
#     return np.array(l)
#
#
#
# def load_data():
#     n_samples = 100
#     positive_X = generate_data(a=10, b=20, n=n_samples)
#     negative_X = generate_data(a=0, b=20, n=n_samples)
#     X = np.concatenate((positive_X, negative_X))
#     y = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))
#     return X, y


def pca_3(X: np.ndarray, y: np.ndarray, to_show: bool = True, to_save: bool = False):
    """
    https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
    https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
    """
    print('PCA 3')
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X_reduced = pca.transform(X)

    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    if to_show:
        plt.show()
    if to_save:
        plt.savefig('plot/pca_3.png')


def pca_2(X: np.ndarray, y: np.ndarray, to_show: bool = True, to_save: bool = False):
    """
    https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py
    https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
    """
    print('PCA 2')
    pca = decomposition.PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    colors = ['red', 'green']
    names = ['evil twin', 'normal']
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [1, 0], names):
        plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
                    color=color, lw=2, label=target_name)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    # plt.axis([-4, 4, -1.5, 1.5])

    if to_show:
        plt.show()
    if to_save:
        plt.savefig('plot/pca_2.png')


def pca_2_incremental(X: np.ndarray, y: np.ndarray, to_show: bool = True, to_save: bool = False):
    """
    https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py
    https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
    """
    print('PCA 2 incemental')
    ipca = IncrementalPCA(n_components=2, batch_size=10)
    X_reduced = ipca.fit_transform(X)

    colors = ['red', 'green']
    names = ['evil twin', 'normal']
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [1, 0], names):
        plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
                    color=color, lw=2, label=target_name)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    # plt.axis([-4, 4, -1.5, 1.5])

    if to_show:
        plt.show()
    if to_save:
        plt.savefig('plot/pca_2_incremental.png')


def t_sne_2(X: np.ndarray, y: np.ndarray, to_show: bool = True, to_save: bool = False):
    print('TSNE 2')
    colors = ['red', 'green']
    names = ['evil twin', 'normal']

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)

    target_ids = range(len(names))
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, names):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()

    if to_show:
        plt.show()
    if to_save:
        plt.savefig('plot/t_sne_2.png')

@numba.njit()
def red_channel_dist(a,b):
    return np.abs(a[0] - b[0])


@numba.njit()
def hue(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if cmax == r:
        return ((g - b) / delta) % 6
    elif cmax == g:
        return ((b - r) / delta) + 2
    else:
        return ((r - g) / delta) + 4

@numba.njit()
def lightness(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    return (cmax + cmin) / 2.0

@numba.njit()
def saturation(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    chroma = cmax - cmin
    light = lightness(r, g, b)
    if light == 1:
        return 0
    else:
        return chroma / (1 - abs(2*light - 1))

@numba.njit()
def hue_dist(a, b):
    diff = (hue(a[0], a[1], a[2]) - hue(b[0], b[1], b[2])) % 6
    if diff < 0:
        return diff + 6
    else:
        return diff

@numba.njit()
def sl_dist(a, b):
    a_sat = saturation(a[0], a[1], a[2])
    b_sat = saturation(b[0], b[1], b[2])
    a_light = lightness(a[0], a[1], a[2])
    b_light = lightness(b[0], b[1], b[2])
    return (a_sat - b_sat)**2 + (a_light - b_light)**2

@numba.njit()
def hsl_dist(a, b):
    a_sat = saturation(a[0], a[1], a[2])
    b_sat = saturation(b[0], b[1], b[2])
    a_light = lightness(a[0], a[1], a[2])
    b_light = lightness(b[0], b[1], b[2])
    a_hue = hue(a[0], a[1], a[2])
    b_hue = hue(b[0], b[1], b[2])
    return (a_sat - b_sat)**2 + (a_light - b_light)**2 + (((a_hue - b_hue) % 6) / 6.0)


def umap_2(X: np.ndarray, y: np.ndarray, title: str, to_show: bool = True, to_save: bool = False,  metric='euclidean',
           n_neighbors=15, min_dist=0.1):
    print('UMAP 2')
    names = ['unsafe', 'legitimate']
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

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9)
    plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()


def main(to_show: bool = True, to_save: bool = False, normalize=True, standardize=True):
    print('norm: {}'.format(normalize))
    print('stand: {}'.format(standardize))

    # os.mkdir('plot')

    positive_data_path = 'data/all_positive_X.txt'
    negative_data_path = 'data/all_negative_X.txt'
    # data: Tuple[np.ndarray, np.ndarray] = prepare_data(positive_data_path, negative_data_path, normalize=normalize,
    #                                                    standardize=standardize)
    #
    # X = data[0]
    # y = data[1]

    # X, y = load_data()

    # pca_3(X, y, to_show)
    # pca_2(X, y, to_show=to_show, to_save=to_save)
    # pca_2_incremental(X, y, to_show=to_show, to_save=to_save)
    # t_sne_2(X, y, to_show=to_show, to_save=to_save)
    index = 1

    # n_neighbors = 200
    # d = 0.8
    # m = "euclidean"

    n_neighbors = 15
    d = 0.1
    m = "euclidean"

    ts = time.time()
    print('iteration: {}'.format(index))
    data: Tuple[np.ndarray, np.ndarray] = prepare_data(positive_data_path, negative_data_path,
                                                                   normalize=normalize,
                                                                   standardize=standardize)

    X = data[0]
    y = data[1]
    print('     << data ready')
    name = m if type(m) is str else m.__name__
    # title = 'UMAP_{}_{}_{}'.format(name, n_neighbors, d)
    title = 'UMAP - Unsafe and Legitimate websites'.format(name, n_neighbors, d)
    umap_2(X, y, title=title, to_show=to_show, to_save=to_save,  metric=m, min_dist=d, n_neighbors=n_neighbors)
    print('     << done in {}'.format(time.time() - ts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", help="If true, show plots_norm_stand.", action='store_true')
    parser.add_argument("--save", help="If true. save plots_norm_stand.", action='store_true')
    parser.add_argument("--no_norm", help="If true. save plots_norm_stand.", action='store_true')
    parser.add_argument("--no_stand", help="If true. save plots_norm_stand.", action='store_true')
    args = parser.parse_args()

    to_show = args.show
    to_save = args.save
    no_norm = args.no_norm
    no_stand = args.no_stand

    norm = True
    if bool(no_norm) is True:
        norm = False
    stand = True
    if bool(no_stand) is True:
        stand = False

    # main(to_show, to_save, norm, stand)
    main(True, False, False, False)



# from sklearn import datasets
#
# np.random.seed(5)
#
# centers = [[1, 1], [-1, -1], [1, -1]]
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
#
#
# print(y.shape)
# print(y)
#
#
# y = np.choose(y, [1, 2, 0]).astype(np.float)
# print(y.shape)
# print(y)