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