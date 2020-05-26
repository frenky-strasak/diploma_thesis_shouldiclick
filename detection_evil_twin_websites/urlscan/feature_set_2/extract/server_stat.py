import numpy as np

from typing import Dict


def get_unixtime(time_string: str) -> float:
    tmp = np.datetime64(time_string)
    unixtime = tmp.view('<i8') / 1e3
    return unixtime


class Server_stat:

    def __init__(self, urlscan_dict: Dict):
        self.data_available = True
        self.task_time = urlscan_dict['task']['time']
        try:
            self.serverStats = urlscan_dict['stats']['serverStats']
        except KeyError:
            self.data_available = False

        self.json_path = 'data__cookies__'
        self.categorical_features_names = [
        ]

        self.numerical_features_names = [
            'count',
            'size',
            'ips',
        ]

        self.absolute_features_names = [
            "len",
        ]

        self.categorical_dict = {}
        for feature_name in self.categorical_features_names:
            self.categorical_dict[feature_name] = {}

        self.numerical_dict = {}
        self.help_full_list_dict = {}
        for feature_name in self.numerical_features_names:
            self.help_full_list_dict[feature_name] = []
            self.numerical_dict[feature_name] = {}

        self.absolute_dict = {}
        for feature_name in self.absolute_features_names:
            self.absolute_dict[feature_name] = 0
        self.absolute_dict['len'] = len(self.serverStats)

    def compute_features(self):
        if self.data_available:
            for stat in self.serverStats:

                for category_feature in self.categorical_features_names:
                    try:
                        res = stat[category_feature]
                    except KeyError:
                        res = 'None'

                    if self.categorical_dict[category_feature].get(res, None) is None:
                        self.categorical_dict[category_feature][res] = 1
                    else:
                        self.categorical_dict[category_feature][res] += 1


                for numerical_feature in self.numerical_features_names:
                    if numerical_feature == 'ips':
                        try:
                            res = len(stat[numerical_feature])
                        except KeyError:
                            res = 0.0
                    else:
                        try:
                            res = float(stat[numerical_feature])
                        except KeyError:
                            res = 0.0
                    self.help_full_list_dict[numerical_feature].append(res)

            self.__compute_4_tuple()


    def __compute_4_tuple(self):
        for numerical_feature in self.numerical_features_names:
            if len(self.help_full_list_dict[numerical_feature]) > 0:
                np_arr = np.array(self.help_full_list_dict[numerical_feature])
                self.numerical_dict[numerical_feature]['mean'] = float(np_arr.mean())
                self.numerical_dict[numerical_feature]['std'] = float(np_arr.std())
                self.numerical_dict[numerical_feature]['max'] = float(np_arr.max())
                self.numerical_dict[numerical_feature]['min'] = float(np_arr.min())
            else:
                self.numerical_dict[numerical_feature]['mean'] = 0.0
                self.numerical_dict[numerical_feature]['std'] = 0.0
                self.numerical_dict[numerical_feature]['max'] = 0.0
                self.numerical_dict[numerical_feature]['min'] = 0.0


    def get_all_feature_list(self) -> list:
        return self.categorical_features_names + self.numerical_features_names + self.absolute_features_names