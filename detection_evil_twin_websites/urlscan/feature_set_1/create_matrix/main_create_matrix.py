import json
import argparse

from feature_extraction.feature_vector import get_feature_vector
from choose_samples import choose_samples
from typing import Tuple

def __get_computed_vector(json_path: str) -> Tuple[list, int]:
    try:
        json_file = open(json_path)
    except (FileNotFoundError, FileExistsError):
        return [], 1

    try:
        json_dict = json.loads(json_file.read())
    except json.decoder.JSONDecodeError:
        return [], 2

    succ, sample_vector, err = get_feature_vector(json_data=json_dict)
    if succ:
        return sample_vector, err
    else:
        return [], err


def __get_uuid_from_path(path: str) -> str:
    if path[-1] == '/':
        name = path.split('/')[-3]
    else:
        name = path.split('/')[-2]
    uuid = name.split('_')[-1]
    return uuid


def __features_to_file(feature_vectors_list: list, output_file: str):
    print('Saving to the file.')
    with open(output_file, 'w') as f:
        for uuid, features in feature_vectors_list:
            line = ' '.join([str(e) for e in features])
            f.write(uuid + ' ' + line + '\n')


def __errors_to_file(dict_err: dict, output_file: str) -> None:
    core_name = output_file.split('_')[0]
    with open(core_name + '_err_json.txt', 'w') as f:
        for err, sample_list in dict_err.items():
            for sample_path in sample_list:
                f.write('{} {}\n'.format(err, sample_path))


def process_err(dict_err: dict, err: int, sample_path) -> None:
    if err != 0:
        if dict_err.get(err, None) is None:
            dict_err[err] = [sample_path]
        else:
            dict_err[err].append(sample_path)


def __process_urlscan(sample_list: list, output_file: str):
    """
    1. Get feature vector from urlscan json.
    2. Save to the file.
    """
    feature_vectors_list = []
    err_dict = {}
    for i, sample_path in enumerate(sample_list):
        if i % 500 == 0:
            print('{} {}'.format(output_file, i))
        uuid = __get_uuid_from_path(sample_path)
        json_path = sample_path + '/' + uuid + '.json'
        feature_vector, err_int = __get_computed_vector(json_path)
        if feature_vector:
            feature_vectors_list.append((uuid, feature_vector))

        process_err(err_dict, err_int, sample_path)

    __features_to_file(feature_vectors_list, output_file)
    __errors_to_file(err_dict, output_file)


def main(evil_twin_file_path: str, normal_path_file_path: str):
    evil_twin_path_valid_list, normal_path_valid_list = choose_samples(evil_twin_file_path, normal_path_file_path)
    __process_urlscan(evil_twin_path_valid_list, 'positive_X.txt')
    __process_urlscan(normal_path_valid_list, 'negative_X.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta_list", help="A path to file where ETA valid samples are.", required=True)
    parser.add_argument("--normal_list", help="A path to file where NORMAL valid samples are.", required=True)
    args = parser.parse_args()

    main(args.eta_list, args.normal_list)
