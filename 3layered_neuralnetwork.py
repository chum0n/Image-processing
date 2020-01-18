import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from common.functions import *

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
    network['Wh'] = np.random.normal(0, sh, (HNODES, INNODES))
    network['bh'] = np.random.normal(0, sh, (HNODES, 1))
    # 出力層の間のWとb
    so = np.sqrt(1/100)
    network['Wo'] = np.random.normal(0, so, (ONODES, HNODES))
    network['bo'] = np.random.normal(0, so, (ONODES, 1))
    return network

# 入力層での入力->出力
def innodes_cal(x):
    return x

# 中間層での入力->出力
def hnodes_cal(yin, network):
    ah = np.dot(yin, network['Wh'].T) + network['bh'].T
    yh = sigmoid(ah)
    return yh

# 後処理、出力層での入力->出力
def onodes_cal(yh, network):
    ao = np.dot(yh, network['Wo'].T) + network['bo'].T
    yo = softmax(ao)
    return yo

mndata = MNIST("/Users/daisuke/le4nn/")
X, Y = mndata.load_testing()
X = np.array(X)
Y = np.array(Y)

print("0~9999までの整数を入力してください")
idx = int(input())
network = init_network()
x = X[idx]
yin = innodes_cal(x)
yh = hnodes_cal(yin, network)
yo = onodes_cal(yh, network)
print("結果")
print(np.argmax(yo))