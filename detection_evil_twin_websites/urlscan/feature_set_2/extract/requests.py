import numpy as np

from typing import Dict


def get_unixtime(time_string: str) -> float:
    tmp = np.datetime64(time_string)
    unixtime = tmp.view('<i8') / 1e3
    return unixtime


class Requests:

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
        try:
            self.page_domain = str(urlscan_dict['page']['domain'])
            # print('pagedomain', self.page_domain)
        except KeyError:
            self.page_domain = ''

        self.json_path = 'data__cookies__'
        self.categorical_features_names = [
            # requests
            'method',
            "mixedContentType",
            "initialPriority",
            "referrerPolicy",

            "Upgrade-Insecure-Requests",
            "Sec-Fetch-User",

            "type",
            "hasUserGesture",

            "documentURL",
            "hasdomain",
            "hasdomain2",

            # # Responses
            "not_same_document",

            "vary",
            "x-frame-options",

            'mimeType',
            "fromPrefetchCache",
            "protocol",

            ":method",
            "pragma",
            "cache-control",
            "sec-fetch-site",
            "sec-fetch-mode",



        ]

        self.responses = [
            # Responses
            "not_same_document",
            "vary",
            "x-frame-options",
            'mimeType',
            ":method",
            "pragma",
            "cache-control",
            "sec-fetch-site",
            "sec-fetch-mode",
            "fromPrefetchCache",
            "protocol",

        ]

        self.numerical_features_names = [
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
        if self.data_available:
            for request in self.requests_list:
                for category_feature in self.categorical_features_names:

                    if category_feature in self.responses:

                        """ RESPONSES """
                        if "vary" == category_feature or "x-frame-options" == category_feature:
                            try:
                                res = request['response']['response']['headers'][category_feature]
                            except KeyError:
                                res = 'None'
                        elif 'mimeType' == category_feature:
                            try:
                                res = request['response']['response'][category_feature]
                                res = res.split('/')[0]
                            except KeyError:
                                res = 'None'
                        elif 'fromPrefetchCache' == category_feature or "protocol" == category_feature:
                            try:
                                res = request['response']['response'][category_feature]
                            except KeyError:
                                res = 'None'
                        elif 'not_same_document' == category_feature:
                            """ response_type """
                            try:
                                response_type = request['response']['type']
                                temp_request_type = request['request']['type']
                            except KeyError:
                                response_type = 'None'
                                temp_request_type = 'None2'
                            if response_type == temp_request_type:
                                res = 'true'
                            else:
                                res = 'false'
                        else:
                            """ 
                            ":method",
                            "pragma",
                            "cache-control",
                            "sec-fetch-site",
                            "'sec-fetch-mode"
                            """
                            try:
                                res = request['response']['response']['requestHeaders'][category_feature]
                            except KeyError:
                                res = 'None'

                    else:
                        """ REQUESTS """
                        if category_feature == "Upgrade-Insecure-Requests" or category_feature == "Sec-Fetch-User":
                            try:
                                res = request['request']['request']['headers'][category_feature]
                            except KeyError:
                                res = 'None'
                        elif "type" == category_feature or "hasUserGesture" == category_feature:
                            try:
                                res = request['request'][category_feature]
                            except KeyError:
                                res = 'None'
                        elif 'documentURL' == category_feature:
                            try:
                                documentURL = request['request']['documentURL']
                                request_url = request['request']['request']['url']
                            except KeyError:
                                return 'None'
                            if documentURL == request_url:
                                res = 'false'
                            else:
                                res = 'true'
                        elif 'hasdomain' == category_feature:
                            try:
                                document_url = str(request['request']['documentURL'])
                                # print('document', document_url)
                                # print('page domain', self.page_domain)
                                if self.page_domain in document_url:
                                    res = 'true'
                                else:
                                    res = 'false'
                            except KeyError:
                                res = 'None'
                        elif 'hasdomain2' == category_feature:
                            try:
                                document_url = request['request']['request']['url']
                                if self.page_domain in document_url:
                                    res = 'true'
                                else:
                                    res = 'false'
                            except KeyError:
                                res = 'None'
                        else:
                            try:
                                res = request['request']['request'][category_feature]
                            except KeyError:
                                res = 'None'


                    assert res is not None, 'Error: res is None'
                    if self.categorical_dict[category_feature].get(res, None) is None:
                        self.categorical_dict[category_feature][res] = 1
                    else:
                        self.categorical_dict[category_feature][res] += 1



                for numerical_feature in self.numerical_features_names:
                    self.help_full_list_dict[numerical_feature].append(res)


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
                for key, value in self.categorical_dict[category_feature].items():
                    self.categorical_dict[category_feature][key] = self.categorical_dict[category_feature][key] / float(count)


    def get_all_feature_list(self) -> list:
        return self.categorical_features_names + self.numerical_features_names + self.absolute_features_names