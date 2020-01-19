import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from common.functions import sigmoid, softmax

INNODES = 784
HNODES = 100
ONODES = 10
PIC_NUM = 10000

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

mndata = MNIST("/Users/daisuke/le4nn/")
X, Y = mndata.load_testing()
X = np.array(X)
Y = np.array(Y)

print("0~9999までの整数を入力してください")
idx = int(input())
x = X[idx]
network = init_network()
y = forward(network, x)
print("一番確率が高いのは", np.argmax(y))

# # 前処理
# def init_network():
#     network = {}
#     # 中間層の間のWとb
#     sh = np.sqrt(1/784)
#     np.random.seed(seed=32)
#     network['Wh'] = np.random.normal(0, sh, (HNODES, INNODES))
#     network['bh'] = np.random.normal(0, sh, (HNODES, 1))
#     # 出力層の間のWとb
#     so = np.sqrt(1/100)
#     network['Wo'] = np.random.normal(0, so, (ONODES, HNODES))
#     network['bo'] = np.random.normal(0, so, (ONODES, 1))
#     return network

# # 入力層での入力->出力
# def innodes_cal(x):
#     return x

# # 中間層での入力->出力
# def hnodes_cal(yin, network):
#     ah = np.dot(yin, network['Wh'].T) + network['bh'].T
#     yh = sigmoid(ah)
#     return yh

# # 後処理、出力層での入力->出力
# def onodes_cal(yh, network):
#     ao = np.dot(yh, network['Wo'].T) + network['bo'].T
#     yo = softmax(ao)
#     return yo

# mndata = MNIST("/Users/daisuke/le4nn/")
# X, Y = mndata.load_testing()
# X = np.array(X)
# Y = np.array(Y)

# print("0~9999までの整数を入力してください")
# idx = int(input())
# network = init_network()
# x = X[idx]
# yin = innodes_cal(x)
# yh = hnodes_cal(yin, network)
# yo = onodes_cal(yh, network)
# print("結果")
# print(np.argmax(yo))