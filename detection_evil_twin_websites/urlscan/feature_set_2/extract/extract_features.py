import argparse
import time
from typing import Dict
import json
import numpy as np

object_list = [
    'cookies',
    'globals',
    'https',
    'requests',
    'console',
    'links',
    'resource_stat',
    'protocol_stat',
    'tls_stat',
    'server_stat',
    'domain_stat',
    'regdomain_stat',
    'ipstat',
    'asn',
    'lists',
]


def __get_uuid_from_path(path: str) -> str:
    if path[-1] == '/':
        name = path.split('/')[-3]
    else:
        name = path.split('/')[-2]
    uuid = name.split('_')[-1]
    return uuid


def get_unixtime(time_string: str) -> float:
    tmp = np.datetime64(time_string)
    unixtime = tmp.view('<i8') / 1e3
    return unixtime


def get_class_name(module_name: str) -> str:
    module_name_list = list(module_name)
    module_name_list[0] = module_name_list[0].upper()
    return ''.join(module_name_list)


def process_samples(sample_file_path: str, output_file: str):

    matrix = []
    ts = time.time()
    with open(sample_file_path) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if line == '' or line is None:
                continue

            uuid = __get_uuid_from_path(line)
            # uuid = 'uuid'
            json_path = line + '/' + uuid + '.json'

            json_file = open(json_path)
            try:
                urlscan_dict: Dict = json.loads(json_file.read())
            except:
                print(i)
                print(line)
                raise json.decoder.JSONDecodeError

            # try:
            #     cookies_list = urlscan_dict['data']['cookies']
            # except KeyError:
            #     continue

            if i % 500 == 0:
                print('{} {}'.format(output_file, i))

            sample = {'uuid': uuid}
            for module_name in object_list:
                module = __import__(module_name)
                class_name = get_class_name(module_name)
                class_ = getattr(module, class_name)
                instance = class_(urlscan_dict)
                instance.compute_features()
                sample[module_name] = {
                    'categorical': instance.categorical_dict,
                    'numerical': instance.numerical_dict,
                    'absolute': instance.absolute_dict,
                }

            matrix.append(sample)

    with open(output_file, 'w') as json_file:
        json.dump(matrix, json_file)


def main(eta_file: str, normal_file: str):
    process_samples(normal_file, 'raw_negative.json')
    process_samples(eta_file, 'raw_positive.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta_list", help="A path to file where ETA valid samples are.", required=True)
    parser.add_argument("--normal_list", help="A path to file where NORMAL valid samples are.", required=True)
    args = parser.parse_args()

    main(args.eta_list, args.normal_list)

