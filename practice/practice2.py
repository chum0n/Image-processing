import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

import sys, os
sys.path.append(os.pardir)
from common.functions import *

INNODES = 784
HNODES = 100
ONODES = 10
BATCH_SIZE = 100

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

def predict(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    # 中間層へ
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1) # (100, 100)
    # 出力層へ
    a2 = np.dot(z1, W2) + b2 # (100, 10)
    y = softmax(a2)

    return y

network = init_network()

mndata = MNIST("../data/mnist")
x_train, t_train = mndata.load_training()
x_train = np.array(x_train) # (60000, 784)
t_train = np.array(t_train) # (60000,)

# ランダムに取り出す
ran_num = np.random.choice(x_train.shape[0], BATCH_SIZE)
x_batch = x_train[ran_num, :] # (100, 784)
t_batch = t_train[ran_num] # (100, )

pre_y = predict(network, x_batch) # (100, 10)
onehot_t_batch = np.eye(10)[t_batch] # (100, 10) 変換元が10種類の場合は、10×10の単位行列を作ってインデックスに変換元の値をいれる

print("cross_entropy_error :", cross_entropy_error(pre_y, onehot_t_batch))