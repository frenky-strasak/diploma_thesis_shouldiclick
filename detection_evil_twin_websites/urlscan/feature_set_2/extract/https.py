import numpy as np

from typing import Dict


def get_unixtime(time_string: str) -> float:
    tmp = np.datetime64(time_string)
    unixtime = tmp.view('<i8') / 1e3
    return unixtime


class Https:

    def __init__(self, urlscan_dict: Dict):
        self.data_available = True
        try:
            self.task_time = urlscan_dict['task']['time']
        except KeyError:
            self.task_time = 0
        try:
            self.requests_list = urlscan_dict['data']['requests']
            self.page_url = urlscan_dict['page']['url']
        except KeyError:
            self.data_available = False
            self.requests_list = []

        self.json_path = 'data__cookies__'
        self.categorical_features_names = [
            "securityState",
            'protocol',
            'keyExchange',
            'keyExchangeGroup',
            'cipher',
            'websitecert'
        ]

        self.numerical_features_names = [
            'sanList'
        ]

        self.absolute_features_names = [
            'len'
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
        self.absolute_dict['len'] = len(self.requests_list)

    def compute_features(self):
        securityState = 'unknown'
        if self.data_available:
            for request in self.requests_list:

                for category_feature in self.categorical_features_names:
                    if category_feature == 'websitecert':
                        continue
                    if category_feature == 'securityState':
                        try:
                            res = request['response']['response'][category_feature]
                        except KeyError:
                            res = 'None'
                    else:
                        try:
                            res = request['response']['response']['securityDetails'][category_feature]
                        except KeyError:
                            res = 'None'

                    if self.categorical_dict[category_feature].get(res, None) is None:
                        self.categorical_dict[category_feature][res] = 1
                    else:
                        self.categorical_dict[category_feature][res] += 1

                if securityState != 'secure':
                    try:
                        url = request['response']['response']['url']
                        if url == self.page_url:
                            securityState = request['response']['response']['securityState']
                    except KeyError:
                        pass

                for numerical_feature in self.numerical_features_names:
                    try:
                        res = request['response']['response']['securityDetails'][numerical_feature]
                        res = len(res)
                    except KeyError:
                        res = 0
                    self.help_full_list_dict[numerical_feature].append(res)


            """ websitecert """
            websitecert_value = 0
            if securityState == 'secure':
                websitecert_value = 1
            self.categorical_dict['websitecert'] = {
                'websitecert': websitecert_value
            }


            self.__compute_4_tuple()
            self.normalize_category()

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


    def normalize_category(self):
        count = self.absolute_dict['len']
        if count > 0:
            for category_feature in self.categorical_features_names:
                if category_feature == 'websitecert':
                    continue
                if category_feature == 'securityState':
                    continue
                for key, value in self.categorical_dict[category_feature].items():
                    self.categorical_dict[category_feature][key] = self.categorical_dict[category_feature][key] / float(count)


    def get_all_feature_list(self) -> list:
        return self.categorical_features_names + self.numerical_features_names + self.absolute_features_names