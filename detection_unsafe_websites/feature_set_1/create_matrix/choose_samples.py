"""
In our beautiful dataset no all samples are valid, because some of them are invalid (404, domain expires, etc...)
For this reason there is (somewhere in your computer) list of valid samples for positive dataset and also for
negative dataset.
The task of this script is to take random samples with the same amount from both datasets (posotove, negative).
"""
import sys
import random
from typing import List, Tuple


def __read_valid_samples(file_path: str) -> list:
    path_list = []
    try:
        with open(file_path) as f:
            for line in f:
                if line[0] == '#':
                    continue
                sample_path = line.rstrip()
                path_list.append(sample_path)
        return path_list
    except (FileExistsError, FileNotFoundError):
        print('The file: {} was not found :( Determining. Bye!'.format(file_path))
        sys.exit(-1)


def __shuffle_lists(evil_twin_path_valid_list: list, normal_path_valid_list: list) -> Tuple[List, List]:
    """Set random seed to have this list same all the time. (Every time when the script will be run,
    the data will be same) """
    seed = 42

    random.seed(seed)
    random.shuffle(evil_twin_path_valid_list)

    random.seed(seed)
    random.shuffle(normal_path_valid_list)
    return evil_twin_path_valid_list, normal_path_valid_list


def __resize(evil_twin_path_valid_list: list, normal_path_valid_list: list) -> Tuple[List, List]:
    min_size = min(len(evil_twin_path_valid_list), len(normal_path_valid_list))
    evil_twin_path_valid_list = evil_twin_path_valid_list[:min_size]
    normal_path_valid_list = normal_path_valid_list[:min_size]
    return evil_twin_path_valid_list, normal_path_valid_list


def choose_samples(evil_twin_file_path: str, normal_path_file_path: str) -> Tuple[List, List]:
    evil_twin_path_valid_list = __read_valid_samples(evil_twin_file_path)
    normal_path_valid_list = __read_valid_samples(normal_path_file_path)

    # evil_twin_path_valid_list, normal_path_valid_list = __shuffle_lists(evil_twin_path_valid_list, normal_path_valid_list)
    # evil_twin_path_valid_list, normal_path_valid_list = __resize(evil_twin_path_valid_list, normal_path_valid_list)

    return evil_twin_path_valid_list, normal_path_valid_list


# if __name__ == '__main__':
    # main()
