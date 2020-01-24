import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from common.functions import sigmoid, softmax
import pickle

INNODES = 3072
HNODES = 100
ONODES = 10

# 前処理
def init_network():
    network = {}
    # 中間層の間のWとb
    network['W1'] = np.load('networkWh.npy')
    network['b1'] = np.load('networkbh.npy')
    # 出力層の間のWとb
    network['W2'] = np.load('networkWo.npy')
    network['b2'] = np.load('networkbo.npy')
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

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    X =np.array(dict[b'data'])
    X =X.reshape((X.shape[0],-1))
    Y =np.array(dict[b'labels'])
    return X,Y

x_test, t_test = unpickle("/Users/daisuke/le4nn/cifar-10-batches-py/test_batch")
x_test = np.array(x_test) # (50000, 3*32*32)
t_test = np.array(t_test) # (50000,)

print("Please enter an integer between 0 and 9999")
idx = int(input())
network = init_network()
x = x_test[idx]
y = forward(network, x)

print("result :",np.argmax(y))
print("The answer is",t_test[idx])

cor = 0
for i in range(10000) :
    x = x_test[i]
    y = forward(network, x)
    if np.argmax(y) == t_test[i] :
        cor += 1
print("test_acc :", cor/10000*100,"%")