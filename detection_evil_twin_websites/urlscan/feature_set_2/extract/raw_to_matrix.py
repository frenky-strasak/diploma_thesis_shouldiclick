from typing import Dict
import json
import numpy as np




all_main_feature_names = [

]







def load_feature_names_and_values(json_path, main_dict: dict, ignored_features: dict):
    json_file = open(json_path)
    raw_matrix = json.loads(json_file.read())

    feature_names = []
    for sample_dict in raw_matrix:
        for data_section_name, feature_dict in sample_dict.items():

            if main_dict.get(data_section_name, None) is None:
                main_dict[data_section_name] = {
                    'categorical': {},
                    'numerical': {},
                    'absolute': {},
                }

            if data_section_name == 'uuid':
                continue
            for feature_type in ['categorical', 'numerical', 'absolute']:
                if feature_type == 'absolute':
                    for feature_name, value in feature_dict[feature_type].items():
                        _ = float(value)
                        # main_feature_name = '__'.join([data_section_name, feature_type, feature_name])
                        # feature_names.append(main_feature_name)
                        main_dict[data_section_name][feature_type][feature_name] = 1
                else:
                    for feature_name, value_dict in feature_dict[feature_type].items():
                        if not isinstance(value_dict, dict):
                            print(feature_name)
                            print(value_dict)
                            raise TypeError
                        for value_name, value in value_dict.items():
                            _ = float(value)
                            # main_feature_name = '__'.join([data_section_name, feature_type, feature_name, value_name])

                            # print(main_feature_name, value)
                            # feature_names.append(main_feature_name)
                            if main_dict[data_section_name][feature_type].get(feature_name, None) is None:
                                main_dict[data_section_name][feature_type][feature_name] = {}
                            main_dict[data_section_name][feature_type][feature_name][value_name] = 1


    # print(len(feature_names))


def compute_feature_value(value: int, amount_of_request: int) -> float:
    try:
        res = value / float(amount_of_request)
    except ZeroDivisionError:
        res = 0.0
    return res


def creeate_feature_matrix(json_path: str, file_name: str, feature_names_list):
    print('loading: {}'.format(json_path))
    json_file = open(json_path)
    raw_matrix = json.loads(json_file.read())

    feature_matrix_list = []
    for sample_dict in raw_matrix:
        # print(sample_dict)
        sample_value_list = []
        for feature_name in feature_names_list:
            try:
                splited_feature = feature_name.split('__')
                temp_d = sample_dict
                for feature_part in splited_feature:
                    x = temp_d[feature_part]
                    if isinstance(x, dict):
                        temp_d = x
                    else:
                        value = float(x)
                        sample_value_list.append(value)
            except KeyError:
                sample_value_list.append(0.0)

        assert len(sample_value_list) == len(feature_names_list)
        feature_matrix_list.append(sample_value_list)
    #
    X = np.array(feature_matrix_list)
    np.save(file_name, X)

def read_feature_names():
    features_names = []
    with open('feature_names_310.txt') as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            features_names.append(line)
    return features_names

def main():
    """ Test of checking features"""
    #
    # feature_dict = {}
    # ignored_features = {}
    # feature_list = []
    #
    # negative_path = 'raw_negative.json'
    # positive_path = 'raw_positive.json'
    # load_feature_names_and_values(positive_path, feature_dict, ignored_features)
    # load_feature_names_and_values(negative_path, feature_dict, ignored_features)
    #
    #
    # for data_section_name, feature_dict in feature_dict.items():
    #     for feature_type in ['categorical', 'numerical', 'absolute']:
    #         if feature_type == 'absolute':
    #             for feature_name, value in feature_dict[feature_type].items():
    #                 main_feature_name = '__'.join([data_section_name, feature_type, feature_name])
    #                 feature_list.append(main_feature_name)
    #         else:
    #             for feature_name, value_dict in feature_dict[feature_type].items():
    #                 if not isinstance(value_dict, dict):
    #                     print(feature_name)
    #                     print(value_dict)
    #                     raise TypeError
    #                 # if len(value_dict.keys()) > 20:
    #                 if len(value_dict.keys()) > 50:
    #                     continue
    #                 for value_name, value in value_dict.items():
    #                     main_feature_name = '__'.join([data_section_name, feature_type, feature_name, value_name])
    #                     if 'vary' in main_feature_name:
    #                         continue
    #                     feature_list.append(main_feature_name)
    #
    #
    #
    # for i, feature_name in enumerate(feature_list):
    #     print(i, feature_name)
    #     # print(feature_name)

    # print(len(all_main_feature_names))
    #

    feature_names = read_feature_names()

    negative_path = 'all_raw_negative.json'
    positive_path = 'all_raw_positive.json'

    creeate_feature_matrix(negative_path, '../splited_data_310/all_negative.npy', feature_names)
    creeate_feature_matrix(positive_path, '../splited_data_310/all_positive.npy', feature_names)


if __name__ == '__main__':
    main()


