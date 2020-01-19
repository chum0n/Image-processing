import numpy as np
from functions import *

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x): # 順伝播
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout): # 逆伝播
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # mask変数はxの要素が0以下の場所をTrue,それ以外をFalse
        out = x.copy() # 一旦入ってきたやつをコピー
        out[self.mask] = 0 # Trueになってるところを0に

        return out

    def backward(self, dout):
        dout[self.mask] = 0 # Trueになってるところを0に
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        # self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # # テンソル対応
        # self.original_x_shape = x.shape
        # x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

# softmax関数とクロスエントロピー誤差
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None # softmaxの出力
        self.t = None # 教師データ(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size # 割ることでデータ１個あたりの誤差が前レイヤへ伝播する
        # if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
        #     dx = (self.y - self.t) / batch_size
        # else:
        #     dx = self.y.copy()
        #     dx[np.arange(batch_size), self.t] -= 1
        #     dx = dx / batch_size

        return dx