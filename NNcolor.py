import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from LayerNetFor3 import LayerNet
from common.optimizer import *
import pickle

INNODES = 784
HNODES = 100
ONODES = 10

ITER_NUM = 50000 # 勾配法による更新の回数
TEACH_NUM = 60000 # 教師データの数
BATCH_SIZE = 100
LEARNING_LATE = 0.01
ITER_PER_EPOC = max(TEACH_NUM / BATCH_SIZE, 1)

network = LayerNet(INNODES, HNODES, ONODES)
optimizer = SGD(lr = LEARNING_LATE)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    X =np.array(dict[b'data'])
    X =X.reshape((X.shape[0],3,-1))
    Y =np.array(dict[b'labels'])
    return X,Y

X,Y = unpickle("/Users/daisuke/le4nn/cifar-10-batches-py/data_batch_1")
x = np.array(X) # (10000, 3, 32, 32)
y = np.array(Y) # (10000,)
print(x.shape)
print(y.shape)

for i in range(ITER_NUM):
    ran_num = np.random.choice(x.shape[0], BATCH_SIZE)
    x_batch = x[ran_num, :] # (100, 784)
    y_batch = y[ran_num] # (100, )
    onehot_y_batch = np.eye(10)[y_batch] # (100, 10) 変換元が10種類の場合は、10×10の単位行列を作ってインデックスに変換元の値をいれる

    # 誤差逆伝播法によって勾配を求める
    grads = network.gradient(x_batch, onehot_y_batch)

    # 更新
    optimizer.update(network.params, grads)

    loss = network.loss(x_batch, onehot_y_batch)

    if i % ITER_PER_EPOC == 0: # エポック終了時
        train_acc = network.accuracy(x, y)

        print(int(i / ITER_PER_EPOC) + 1, ": train_acc, cross_entropy_error |", train_acc*100, "%,",loss)

np.save('networkWh',network.params['W1'])
np.save('networkbh',network.params['b1'])
np.save('networkWo',network.params['W2'])
np.save('networkbo',network.params['b2'])