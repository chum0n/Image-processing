import numpy as np
from common.functions import *
from common.layers import *
from collections import OrderedDict

class LayerNet:

    def __init__(self, INNODES, HNODES, ONODES, gamma, beta):
        # 重みの初期化
        self.params = {}
        # 中間層の間のWとb
        sh = np.sqrt(1/784)
        np.random.seed(seed=32)
        self.params['W1'] = np.random.normal(0, sh, (INNODES, HNODES)) # (784, 100)
        self.params['b1'] = np.random.normal(0, sh, (1, HNODES)) # (1, 100)
        # 出力層の間のWとb
        so = np.sqrt(1/100)
        self.params['W2'] = np.random.normal(0, so, (HNODES, ONODES)) # (100, 10)
        self.params['b2'] = np.random.normal(0, so, (1, ONODES)) # (1, 10)

        # レイヤの生成
        # 順番付きディクショナリ変数
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['BatchNormalization1'] = BatchNormalization(gamma, beta)
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # 追加した順にレイヤのforward呼ぶだけでok
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 損失関数の値を求める、xは画像データ,tは正解ラベル
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # 認識精度を求める
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 重みパラメータに対する勾配を誤差逆伝播法で求める
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
