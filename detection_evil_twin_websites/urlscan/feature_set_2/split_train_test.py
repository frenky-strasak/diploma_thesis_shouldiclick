import random
import numpy as np

from sklearn.model_selection import train_test_split


def load_data(path: str):

    samples = np.load(path)

    np.random.shuffle(samples)

    assert samples.shape[0] == 10194

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

    np.save('train_' + path, x_train)
    np.save('test_' + path, x_test)

    return samples.shape[0]



neg = load_data('negative.npy')
pos = load_data('positive.npy')


print(pos)
print(neg)

