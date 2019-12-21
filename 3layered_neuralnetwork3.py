import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

INNODES = 784
HNODES = 100
ONODES = 10
PIC_NUM = 10000
TEACH_NUM = 60000
BATCH = 100
ETA = 0.01
EPOC = int(TEACH_NUM/BATCH)

# 前処理
def init_network():
    network = {}
    # 中間層の間のWとb
    sh = np.sqrt(1/784)
    #np.random.seed(seed=32)
    network['Wh'] = np.random.normal(0, sh, (HNODES, INNODES))
    network['bh'] = np.random.normal(0, sh, (HNODES))
    # 出力層の間のWとb
    so = np.sqrt(1/100)
    network['Wo'] = np.random.normal(0, so, (ONODES, HNODES))
    network['bo'] = np.random.normal(0, so, (ONODES))
    return network

# 入力層での入力->出力
def innodes_cal(x):
    return x

# シグモイド関数(活性化関数)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 中間層での入力->出力
def hnodes_cal(yin, network):
    ah = np.dot(yin, network['Wh'].T) + network['bh'].T
    yh = sigmoid(ah)
    return yh

# ソフトマックス関数
def softmax(a):
    # 一番大きい値を取得
    c = np.max(a, axis=1, keepdims=True)
    # 各要素から一番大きな値を引く（オーバーフロー対策）
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    # 要素の値/全体の要素の合計
    y = exp_a / sum_exp_a

    return y

# 後処理、出力層での入力->出力
def onodes_cal(yh, network):
    ao = np.dot(yh, network['Wo'].T) + network['bo'].T
    yo = softmax(ao)
    return yo

# エントロピー誤差
def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

# ミニバッチ
def mini_batch(e) :
    En = np.sum(e)/BATCH
    return En

# 逆伝搬のシグモイド関数
def resigmoid(x):
    return (1 - x) * x

mndata = MNIST("/Users/daisuke/le4nn/")
X, Y = mndata.load_training()
X = np.array(X)
Y = np.array(Y)
network = init_network()

for re in range(50) :
    for i in range(EPOC) :
        ran_num = np.random.choice(X.shape[0], BATCH)
        x = X[ran_num, :]
        y = Y[ran_num]
        yin = innodes_cal(x)
        yh = hnodes_cal(yin, network)
        yo = onodes_cal(yh, network)
        # onehot
        hotY = np.eye(10)[y]

        # 処理手順４
        slopeAk = (yo - hotY) / BATCH
        # 転置する
        slopeAk = slopeAk.T

        # 処理手順５
        slopeX = np.dot(network['Wo'].T, slopeAk)
        slopeW2 = np.dot(slopeAk, yh)
        slopeb2 = np.sum(slopeAk, axis = 1)

        # 処理手順６
        difsig = resigmoid(yh).T * slopeX # 50*100

        # 処理手順７
        slopeX = np.dot(network['Wh'].T, difsig)
        slopeW1 = np.dot(difsig, yin)
        slopeb1 = np.sum(difsig, axis = 1)

        # 処理手順８
        network['Wh'] = network['Wh'] - ETA * slopeW1
        network['Wo'] = network['Wo'] - ETA * slopeW2
        network['bh'] = network['bh'] - ETA * slopeb1
        network['bo'] = network['bo'] - ETA * slopeb2

    print("クロスエントロピー誤差",re + 1,"回目")
    print(mini_batch(cross_entropy(yo, hotY)))

np.save('networkWh',network['Wh'])
np.save('networkWo',network['Wo'])
np.save('networkbh',network['bh'])
np.save('networkbo',network['bo'])

