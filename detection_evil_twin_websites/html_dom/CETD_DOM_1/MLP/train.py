import matplotlib.pyplot as plt
import argparse

import os
import re
import datetime

from dataset import Dataset
from cifar_net import CifarNet
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--from_file", default=None, type=str, help="Specify if you want weights from a file")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")

parser.add_argument("--augment", default=False, type=bool, help="Augment data.")
args = parser.parse_args()

args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))




print("=========================================================================================")
print("Loading data...")
dataset = Dataset()
print("Done.")
print("=========================================================================================")
# Data info
print("=========================================================================================")
print("Data info")
print("*****************************************************************************************")
print("Training set size: {}".format(dataset.train.size))
print("*****************************************************************************************")
print("*****************************************************************************************")
print("Dev set size: {}".format(dataset.dev.size))
print("*****************************************************************************************")
print("Test set size: {}".format(dataset.test.size))
print("=========================================================================================")
print("Show first 3 training images")
# for i in range(3):
#     plt.imshow(dataset.train.data['images'][i])
#     plt.show()

cifar_net = CifarNet(dataset, args)

cifar_net.train()

predcited = []
for probs in cifar_net.predict(dataset.test.data["images"], batch_size=args.batch_size):
    pred = np.argmax(probs)
    predcited.append(pred)



accuracy_test = accuracy_score(dataset.test.data["labels"], predcited)
TN, FP, FN, TP = confusion_matrix(dataset.test.data["labels"], predcited).ravel()

_test_acc = (TP + TN) / (FP + TN + FN + TP)
print('_test_acc: {}'.format(_test_acc))
# FPR = FP / (FP + TN)
FPR = FP / (FP + TN)
print('False Positive Rate FPR: {}'.format(FPR))
# FDR = FP / (FP + TP)
FDR = FP / (FP + TP)
print('False Discovery Rate FDR: {}'.format(FDR))
# FNR = FN / (FN + TP)
FNR = FN / (FN + TP)
print('False Negative Rate FNR: {}'.format(FNR))
# TPR = TP / (TP + FN)
TPR = TP / (TP + FN)
print('Sensitivity TPR: {}'.format(TPR))
# SPC = TN / (FP + TN)
SPC = TN / (FP + TN)
print('Specificity SPC: {}'.format(SPC))
# PPV = TP / (TP + FP)
PPV = TP / (TP + FP)
print('Precision PPV: {}'.format(PPV))
# NPV = TN / (TN + FN)
NPV = TN / (TN + FN)
print('Negative Predictive Value NPV: {}'.format(NPV))
