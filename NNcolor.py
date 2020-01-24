import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from LayerNetFor3 import LayerNet
from common.optimizer import *
import pickle

INNODES = 3072
HNODES = 100
ONODES = 10

ITER_NUM = 20000 # 勾配法による更新の回数
TEACH_NUM = 60000 # 教師データの数
BATCH_SIZE = 100
LEARNING_LATE = 0.01
ITER_PER_EPOC = max(TEACH_NUM / BATCH_SIZE, 1)

network = LayerNet(INNODES, HNODES, ONODES)
optimizer = SGD(lr = LEARNING_LATE)

def unpickleall():
    for i in range(5):
        with open("/Users/daisuke/le4nn/cifar-10-batches-py/data_batch_"+str(i+1), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        file_X = np.array(dict[b'data'])
        file_X = file_X.reshape((file_X.shape[0], -1))
        file_X = file_X / 255.0 # 0から1までの範囲にする
        file_Y = np.array(dict[b'labels'])
        if i == 0:
            X = file_X
            Y = file_Y
        else:
            X = np.append(X, file_X, axis=0)
            Y = np.append(Y, file_Y, axis=0)
    return X, Y

X,Y = unpickleall()
x = np.array(X) # (50000, 3*32*32)
y = np.array(Y) # (50000,)

for i in range(ITER_NUM):
    ran_num = np.random.choice(x.shape[0], BATCH_SIZE)
    x_batch = x[ran_num, :] # (100, 3*32*32)
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