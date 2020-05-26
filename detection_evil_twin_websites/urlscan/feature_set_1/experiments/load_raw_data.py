"""
This file loads raw feature vectors (computed from urlscan json files) and prepare for ML stuff.
"""
import sys
import numpy as np
from typing import List, Tuple


def __load_feature_names() -> list:
    feature_names = []
    with open('feature_names.txt') as f:
        for line in f:
            feature_name = line.rstrip()
            if line == '' or line[0] == '#':
                continue
            feature_names.append(feature_name)
    return feature_names


def __load_removed_feature_names() -> list:
    feature_names = []
    with open('removed_features.txt') as f:
        for line in f:
            feature_name = line.rstrip()
            if line == '' or line[0] == '#':
                continue
            feature_names.append(feature_name)
    return feature_names


def apply_filter(feature_name: str, features_removed_names: list) -> bool:
    # for removed_feature in features_removed_names:
    #     if feature_name in removed_feature or removed_feature in feature_name:
    #         return False
    # return True
    if feature_name in features_removed_names:
        return False
    return True


def __load_data_file(file_path: str, normalize: bool = False) -> List[List[float]]:
    features_names = __load_feature_names()
    features_removed_names = __load_removed_feature_names()
    try:
        feature_vector_list = []
        with open(file_path) as f:
            for line in f:
                if line[0] == '#':
                    continue
                if line == '\n':
                    continue
                if line is None:
                    continue
                feture_vector_str = line.rstrip()
                feature_numbers = feture_vector_str.split(' ')[1:]
                feature_list = []
                for value, feature_name in zip(feature_numbers, features_names):

                    if float('nan') == float(value):
                        value = 0.0

                    if float('inf') == float(value):
                        value = 0.0

                    if normalize:
                        if float(value) < 0:
                            value = 0.0

                    if apply_filter(feature_name, features_removed_names):
                        feature_list.append(float(value))

                assert len(feature_list) == len(feature_numbers) - len(features_removed_names), \
                    'Final features: {}, all feature on line: {}, removed_features: {}'.format(len(feature_list), len(feature_numbers), len(features_removed_names))
                feature_vector_list.append(feature_list)
        return feature_vector_list
    except (FileNotFoundError, FileExistsError):
        print('Error: wrong path. Terminating.')
        sys.exit(-1)


def __data_to_numpy(feature_vector_list: List[List[float]]) -> np.ndarray:
    samples_amount = len(feature_vector_list)
    features_amount = len(feature_vector_list[0])
    feature_matrix = np.array(feature_vector_list)
    assert feature_matrix.shape[0] == samples_amount
    assert feature_matrix.shape[1] == features_amount
    return feature_matrix


# def __cretae_labels(min_length: int) -> np.ndarray:
#     p = np.ones(min_length)
#     n = np.zeros(min_length)
#     y = np.concatenate((p, n))
#     return y

def __cretae_labels(positive_length: int, negative_path: int) -> np.ndarray:
    p = np.ones(positive_length)
    n = np.zeros(negative_path)
    y = np.concatenate((p, n))
    return y


def __standardize(feature_matrix: np.ndarray) -> np.ndarray:
    for i in range(feature_matrix.shape[1]):
        mean = feature_matrix[:, i].mean()
        std = feature_matrix[:, i].std()
        if std != 0:
            feature_matrix[:, i] = (feature_matrix[:, i] - mean) / std
        else:
            feature_matrix[:, i] = (feature_matrix[:, i] - mean)
    return feature_matrix


def __normalize(feature_matrix: np.ndarray) -> np.ndarray:
    for i in range(feature_matrix.shape[1]):
        max_value = feature_matrix[:, i].max()
        if max_value != 0:
            feature_matrix[:, i] = feature_matrix[:, i] / max_value
        return feature_matrix


def prepare_data(positive_data_path: str, negative_data_path: str, normalize=False, standardize=False) -> Tuple[np.ndarray, np.ndarray]:
    positive_data__list = __load_data_file(positive_data_path, normalize)
    negative_data__list = __load_data_file(negative_data_path, normalize)

    # all_samples_from_list = len(positive_data__list) + len(negative_data__list)
    min_length = min(len(positive_data__list), len(negative_data__list))
    positive_data__list = positive_data__list[:min_length]
    negative_data__list = negative_data__list[:min_length]

    y = __cretae_labels(len(positive_data__list), len(negative_data__list))

    matrix_list = positive_data__list + negative_data__list
    X = __data_to_numpy(matrix_list)

    assert X.shape[0] == y.shape[0]
    # assert X.shape[0] == all_samples_from_list and y.shape[0] == all_samples_from_list
    if normalize:
        print('normalization')
        X = __normalize(X)

    if standardize:
        print('standartization')
        X = __standardize(X)
    return X, y


# def prepare_data_2(positive_data_path: str, negative_data_path: str, normalize=True) -> Tuple[np.ndarray, np.ndarray]:
#     positive_data__list = __load_data_file(positive_data_path)
#     negative_data__list = __load_data_file(negative_data_path)
#     all_samples_from_list = len(positive_data__list) + len(negative_data__list)
#     # min_length = min(len(positive_data__list), len(negative_data__list))
#     # positive_data__list = positive_data__list[:min_length]
#     # negative_data__list = negative_data__list[:min_length]
#
#     # y = __cretae_labels(len(positive_data__list), len(negative_data__list))
#
#     # matrix_list = positive_data__list + negative_data__list
#     # X = __data_to_numpy(matrix_list)
#     X_pos = __data_to_numpy(positive_data__list)
#     X_neg = __data_to_numpy(negative_data__list)
#
#
#     assert X_pos.shape[0] == len(positive_data__list)
#     assert X_neg.shape[0] == len(negative_data__list)
#
#     if normalize:
#         X = __normalize_2(X_pos, X_neg)
#
#     idx = np.random.permutation(X_pos.shape[0])
#     X_pos = X_pos[idx]
#
#     idx = np.random.permutation(X_neg.shape[0])
#     X_neg = X_neg[idx]
#     return X_pos, X_neg,


def get_feature_names_and_indexes() -> list:
    """
    Return feature names with indexes. The features are not same because some of the were classified as
    "wrong" in 'removed_features.txt'.
    """
    feature_names = __load_feature_names()
    removed_feature_names = __load_removed_feature_names()
    f = []
    index = 0

    """ check """
    for remove_feature in removed_feature_names:
        assert remove_feature in feature_names

    for feature_name in feature_names:
        if feature_name not in removed_feature_names:
            f.append((feature_name, index))
            index += 1
    return f
