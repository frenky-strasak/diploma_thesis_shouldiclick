import numpy as np

from typing import Dict


def get_unixtime(time_string: str) -> float:
    tmp = np.datetime64(time_string)
    unixtime = tmp.view('<i8') / 1e3
    return unixtime


class Lists:

    def __init__(self, urlscan_dict: Dict):
        self.data_available = True
        self.task_time = urlscan_dict['task']['time']
        try:
            self.lists = urlscan_dict['lists']
        except KeyError:
            self.data_available = False

        self.json_path = 'data__cookies__'
        self.categorical_features_names = [
        ]

        self.numerical_features_names = [
            'sub_domains',
            'url_path_length',
            'url_length',
            'number_counts',

        ]

        self.absolute_features_names = [
            "ips",
            "countries",
            "asns",
            "domains",
            "servers",
            "urls",
            "linkDomains",
            "certificates",
            "hashes",
            "servers",

            'js',
            'img',
            'css',
            'cookie',
            '?',
            'html',
            'dll',
            '@',
            '//',
            '=',
            '-',

        ]
        self.special_chars = [
            'js',
            'img',
            'css',
            'cookie',
            '?',
            'html',
            'dll',
            '@',
            '//',
            '=',
            '-',
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
            if feature_name not in self.special_chars:
                try:
                    self.absolute_dict[feature_name] = len(self.lists[feature_name])
                except KeyError:
                    self.absolute_dict[feature_name] = 0

        js_urls = 0
        try:
            for url in self.lists['urls']:
                if '.js' in url:
                    js_urls += 1
        except KeyError:
            pass
        self.absolute_dict['js'] = js_urls

        img = 0
        try:
            for url in self.lists['urls']:
                url_lower = url.lower()
                for ext in ['.png', '.jpg', 'jpeg', 'gif', '.svg']:
                    if ext in url_lower:
                        img += 1
                        break
        except KeyError:
            pass
        self.absolute_dict['img'] = img

        css = 0
        try:
            for url in self.lists['urls']:
                if '.css' in url:
                    css += 1
        except KeyError:
            pass
        self.absolute_dict['css'] = css

        cookie = 0
        try:
            for url in self.lists['urls']:
                if 'cookie' in url:
                    cookie += 1
        except KeyError:
            pass
        self.absolute_dict['cookie'] = cookie

        question_mark = 0
        try:
            for url in self.lists['urls']:
                k = len(url.split('?'))
                question_mark += k
        except KeyError:
            pass
        self.absolute_dict['?'] = question_mark

        html = 0
        try:
            for url in self.lists['urls']:
                if '.html' in url:
                    html += 1
        except KeyError:
            pass
        self.absolute_dict['html'] = html

        dll = 0
        try:
            for url in self.lists['urls']:
                if '.dll' in url:
                    dll += 1
        except KeyError:
            pass
        self.absolute_dict['dll'] = dll

        at_sign = 0
        try:
            for url in self.lists['urls']:
                k = len(url.split('@'))
                at_sign += k
        except KeyError:
            pass
        self.absolute_dict['@'] = at_sign

        two_backslaches = 0
        try:
            for url in self.lists['urls']:
                k = len(url.split('//'))
                two_backslaches += k
        except KeyError:
            pass
        self.absolute_dict['//'] = two_backslaches

        equals = 0
        try:
            for url in self.lists['urls']:
                k = len(url.split('='))
                equals += k
        except KeyError:
            pass
        self.absolute_dict['='] = equals

        minus = 0
        try:
            for url in self.lists['urls']:
                k = len(url.split('-'))
                minus += k
        except KeyError:
            pass
        self.absolute_dict['-'] = minus

        under_slash = 0
        try:
            for url in self.lists['urls']:
                k = len(url.split('_'))
                under_slash += k
        except KeyError:
            pass
        self.absolute_dict['_'] = under_slash

    def compute_features(self):
        if self.data_available:

            for numerical_feature in self.numerical_features_names:
                if numerical_feature == 'sub_domains':
                    try:
                        for domain in self.lists['domains']:
                            subdomains = len(str(domain).split('.'))
                            self.help_full_list_dict[numerical_feature].append(subdomains)
                    except (KeyError):
                        self.help_full_list_dict[numerical_feature].append(0)

                elif numerical_feature == 'url_path_length':
                    try:
                        for url in self.lists['urls']:
                            path_len = len(url.split('/'))
                            self.help_full_list_dict[numerical_feature].append(path_len)
                    except KeyError:
                        self.help_full_list_dict[numerical_feature].append(0)
                elif numerical_feature == 'url_length':
                    try:
                        for url in self.lists['urls']:
                            url_len = len(url)
                            self.help_full_list_dict[numerical_feature].append(url_len)
                    except KeyError:
                        self.help_full_list_dict[numerical_feature].append(0)
                elif numerical_feature == 'number_counts':
                    try:
                        for url in self.lists['urls']:
                            digits = 0
                            for ch in list(url):
                                if ch.isdigit():
                                    digits += 1
                            self.help_full_list_dict[numerical_feature].append(digits)
                    except KeyError:
                        self.help_full_list_dict[numerical_feature].append(0)

            self.__compute_4_tuple()
            self.normalize_absolute()

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


    def normalize_absolute(self):
        count = len(self.lists['urls'])
        if count > 0:
            for absolute_feature in self.special_chars:
                self.absolute_dict[absolute_feature] = self.absolute_dict[absolute_feature] / float(count)


    def get_all_feature_list(self) -> list:
        return self.categorical_features_names + self.numerical_features_names + self.absolute_features_names