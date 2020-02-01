import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

import sys, os
sys.path.append(os.pardir)
from common.functions import sigmoid, softmax

INNODES = 784
HNODES = 100
ONODES = 10

# 前処理
def init_network():
    network = {}
    # 中間層の間のWとb
    sh = np.sqrt(1/784)
    np.random.seed(seed=32)
    network['W1'] = np.random.normal(0, sh, (INNODES, HNODES)) # (784, 100)
    network['b1'] = np.random.normal(0, sh, (1, HNODES)) # (1, 100)
    # 出力層の間のWとb
    so = np.sqrt(1/100)
    network['W2'] = np.random.normal(0, so, (HNODES, ONODES)) # (100, 10)
    network['b2'] = np.random.normal(0, so, (1, ONODES)) # (1, 10)
    return network

def forward(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    # 中間層へ
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1) # (1, 100)
    # 出力層へ
    a2 = np.dot(z1, W2) + b2 # (1, 10)
    y = softmax(a2)

    return y

network = init_network()

mndata = MNIST("../data/mnist")
x_test, t_test = mndata.load_testing()
x_test = np.array(x_test)
t_test = np.array(t_test)

print("Please enter an integer between 0 and 9999")
idx = int(input())
x = x_test[idx]
y = forward(network, x)
print("The most likely is", np.argmax(y))