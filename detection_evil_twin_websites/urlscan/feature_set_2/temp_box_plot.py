# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.array([1, 2, 3, 4, 5])
# y = np.power(x, 2) # Effectively y = x**2
# e = np.array([1.5, 2.6, 3.7, 4.6, 5.5])
#
# plt.errorbar(x, y, e, linestyle='None', marker='^')
#
# plt.show()


#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# accruacy_lists = [
#     [0.9321, 0.9234, 0.9355, 0.8999],
#     [0.9212, 0.9199, 0.92587, 0.9245],
#     [0.9234, 0.9192, 0.9257, 0.8948],
# ]

accruacy_lists = [

]


y = [0.9116, 0.8994]
err = [0.0012, 0.0077]
x = [1, 2]

y2 = []
err2 = []
# for i, acc_list in enumerate(accruacy_lists):
#     mean = np.array(acc_list).mean()
#     std = np.array(acc_list).std()
#     y.append(mean)
#     y2.append(mean - 1)
#     err.append(std)
#     err2.append(std - 1)
#     x.append(i+1)


# example data
# x = np.arange(1, 4, 1)
# y = np.exp(-x)
# print(x.shape)


# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
# plt.errorbar(x, y, yerr=err, xlolims=True, label='xlolims=True')

# plot with connected line
# plt.errorbar(x, y, yerr=err, xlolims=True, fmt='--o')
plt.errorbar(x, y, yerr=err, xlolims=True, fmt='o')

# Integer number no float
# matplotlib.pyplot.xticks(x)


# Text instead of numbers
x_names = ['test acc', 'train acc', 'FPR', 'FNR']
# x_names = ['test acc', 'train acc', 'FPR', 'FNR']
plt.xticks(x, x_names)


plt.title("Mean and STD of Accuracy on testing data with K-Fold cross validation")

figure = plt.gcf()  # get current figure
figure.set_size_inches(16, 9)

plt.show()