import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

import sys, os
sys.path.append(os.pardir)
from LayerNet.LN import LayerNet
from common.optimizer import *

INNODES = 784
HNODES = 100
ONODES = 10

ITER_NUM = 18000 # 勾配法による更新の回数
TEACH_NUM = 60000 # 教師データの数
BATCH_SIZE = 100
LEARNING_LATE = 0.01
ITER_PER_EPOC = max(TEACH_NUM / BATCH_SIZE, 1)

network = LayerNet(INNODES, HNODES, ONODES)
optimizer = SGD(lr = LEARNING_LATE)

mndata = MNIST("../data/mnist")
x_train, t_train = mndata.load_training()
x_train = np.array(x_train) # (60000, 784)
t_train = np.array(t_train) # (60000,)

for i in range(ITER_NUM):
    ran_num = np.random.choice(x_train.shape[0], BATCH_SIZE)
    x_batch = x_train[ran_num, :] # (100, 784)
    t_batch = t_train[ran_num] # (100, )
    onehot_t_batch = np.eye(10)[t_batch] # (100, 10) 変換元が10種類の場合は、10×10の単位行列を作ってインデックスに変換元の値をいれる

    # 誤差逆伝播法によって勾配を求める
    grads = network.gradient(x_batch, onehot_t_batch)

    # 更新
    optimizer.update(network.params, grads)

    loss = network.loss(x_batch, onehot_t_batch)

    if i % ITER_PER_EPOC == 0: # エポック終了時
        train_acc = network.accuracy(x_train, t_train)

        print(int(i / ITER_PER_EPOC) + 1, ": train_acc, cross_entropy_error |", train_acc*100, "%,",loss)

np.save('networkWh',network.params['W1'])
np.save('networkbh',network.params['b1'])
np.save('networkWo',network.params['W2'])
np.save('networkbo',network.params['b2'])