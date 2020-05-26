import random

from sklearn.model_selection import train_test_split


def load_data(path: str):
    lines = 0
    samples = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            if line[0] == '#':
                continue
            lines += 1

            samples.append(line)


    random.shuffle(samples)

    assert len(samples) == 10194

    x_train, x_test = train_test_split(samples, test_size=0.2)

    print(len(x_train))
    print(len(x_test))

    # with open('train_' + path, 'w') as f:
    #     for sample in x_train:
    #         f.write(sample + '\n')
    #
    # with open('test_' + path, 'w') as f:
    #     for sample in x_test:
    #         f.write(sample + '\n')

    return lines



neg = load_data('negative_X.txt')
pos = load_data('positive_X.txt')


print(pos)
print(neg)

