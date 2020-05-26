# Team: František Střasák, Daniel Šmolík
# 40419513-4364-11e9-b0fd-00505601122b
# 3f638e58-4364-11e9-b0fd-00505601122b
import os
import sys
import urllib.request

import numpy as np


class Dataset:
    H, W, C = 55, 55, 5
    LABELS = 2

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._data["images"] = self._data["images"].astype(np.float32)
            old_shape = self._data["images"].shape
            self._data["images"] = np.where(self._data["images"] > 255, 255, self._data["images"])
            self._data["images"] = self._data["images"] / 255
            assert old_shape == self._data["images"].shape, 'Error: The shape after the norm is diff!'

            self._size = len(self._data["images"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            while True:
                permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
                while len(permutation):
                    batch_size = min(size or np.inf, len(permutation))
                    batch_perm = permutation[:batch_size]
                    permutation = permutation[batch_size:]

                    batch = (self._data['images'][batch_perm], self._data['labels'][batch_perm])
                    yield batch

    def __init__(self):

        folder = 'data/'
        X_train = np.load(folder + 'X_train.npy')
        y_train = np.load(folder + 'y_train.npy')
        train_dict = {'images': X_train, 'labels': y_train}
        setattr(self, 'train', self.Dataset(train_dict, shuffle_batches=True))


        X_dev = np.load(folder + 'X_val.npy')
        y_dev = np.load(folder + 'y_val.npy')
        dev_dict = {'images': X_dev, 'labels': y_dev}
        setattr(self, 'dev', self.Dataset(dev_dict, shuffle_batches=False))

        X_test = np.load(folder + 'X_test.npy')
        y_test = np.load(folder + 'y_test.npy')
        test_dict = {'images': X_test, 'labels': y_test}
        setattr(self, 'test', self.Dataset(test_dict, shuffle_batches=False))


        # for dataset in ["train", "dev", "test"]:
        #     data = dict((key[len(dataset) + 1:], cifar[key]) for key in cifar if key.startswith(dataset))
        #     setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train"))
