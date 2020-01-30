import numpy as np
from common.functions import *
from common.util import *

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Relu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        out = np.maximum(0, x)

        return out

    def backward(self, dout):
        dx = dout * np.where(self.x > 0, 1, 0)

        return dx

# class Relu:
#     def __init__(self):
#         self.mask = None

#     def forward(self, x):
#         self.mask = (x <= 0) # mask変数はxの要素が0以下の場所をTrue,それ以外をFalse
#         self.x = x
#         out = x.copy() # 一旦入ってきたやつをコピー
#         out[self.mask] = 0 # Trueになってるところを0に

#         return out

#     def backward(self, dout):
#         dout[self.mask] = 0 # Trueになってるところを0に
#         dx = dout

#         return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        # テンソル対応のために用意
        self.x_originalshape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソルに対応する
        self.x_originalshape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # 入力データの形状に戻す
        dx = dx.reshape(*self.x_originalshape)

        return dx

# softmax関数とクロスエントロピー誤差
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # ソフトマックスの出力
        self.t = None # 教師データ(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 割ることでデータ１個あたりの誤差が前レイヤへ伝播する
        dx = (self.y - self.t) / batch_size

        return dx

# Dropout
class Dropout:
    def __init__(self, dropout_ratio=0.5, train_flag=True):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flag = train_flag

    def forward(self, x):
        if self.train_flag:
            # xと同じ形状の配列をランダムに生成し、その値がdropout_ratioよりも大きい要素だけをTrue
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask # ランダムに無視

        else:
            return x * (1.0 - self.dropout_ratio) # 無視しない

    def backward(self, dout): # ReLUと同じ
        dx = dout * self.mask
        return dx

# BatchNormalization
class BatchNormalization:
    def __init__(self, gamma=1, beta=0, momentum=0.9, train_flag=True):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.train_flag = train_flag

        self.batch_size = None
        self.x_miave = None
        self.stadev = None
        self.dgamma = None
        self.dbeta = None

        # 移動平均と移動分散
        self.run_ave = None
        self.run_var = None

    def forward(self, x):
        if self.run_ave is None:
            self.run_ave = np.zeros(x.shape[1])
            self.run_var = np.zeros(x.shape[1])

        if self.train_flag:
            ave = x.mean(axis=0)
            x_miave = x - ave
            var = np.mean(x_miave*x_miave, axis=0)
            stadev = np.sqrt(var + 10e-7)
            x_nor = x_miave / stadev

            self.batch_size = x.shape[0]
            self.x_miave = x_miave
            self.x_nor = x_nor
            self.stadev = stadev
            self.run_ave = self.momentum * self.run_ave + (1-self.momentum) * ave
            self.run_var = self.momentum * self.run_var + (1-self.momentum) * var

            out = self.gamma * x_nor + self.beta

        else:
            out = self.gamma * (x - self.run_ave) / np.sqrt(self.run_var + 10e-7) + self.beta

        return out


    def backward(self, dout):
        dx_nor = dout * self.gamma
        dvar = -0.5 * np.sum((dx_nor * self.x_miave) / (self.stadev * self.stadev * self.stadev), axis=0)
        dave = np.sum(-1 * dx_nor / self.stadev, axis=0) + dvar * -2 * np.sum(self.x_miave, axis=0) / self.batch_size
        dx = dx_nor / self.stadev + (2.0 / self.batch_size) * self.x_miave * dvar + dave / self.batch_size
        dgamma = np.sum(dout * self.x_nor, axis=0)
        dbeta = dout.sum(axis=0)

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

# 畳み込み層
class Convolution:
    def __init__(self, W, b, stride, pad):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.col = None
        self.col_W = None
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x

        # 入ってくるデータのバッチ数、チャンネル、高さ、幅
        N, C, H, W = x.shape
        # フィルターの個数、チャンネル、高さ、幅
        FN, C, FH, FW = self.W.shape
        # 出ていくデータの高さ
        out_h = int((H + 2 * self.pad - FH) / self.stride + 1)
        # 出ていくデータの幅
        out_w = int((W + 2 * self.pad - FW) / self.stride + 1)

        # フィルターによって都合のいいように入力データを展開
        col = im2col(x, out_h, out_w, FH, FW, self.stride, self.pad)
        # フィルターの展開
        col_W = self.W.reshape(FN, -1).T # -1指定で全要素数/FNの形にする
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1)
        # transpose関数は多次元配列の軸の順番を入れ替える関数
        out = out.transpose(0, 3, 1, 2)

        self.col = col
        self.col_W = col_W

        return out

    # 逆伝播はほぼAffineと同じ
    def backward(self, dout):
        # 入ってくるデータのバッチ数、チャンネル、高さ、幅
        N, C, H, W = self.x.shape
        # フィルターの個数、チャンネル、高さ、幅
        FN, C, FH, FW = self.W.shape

        dout = dout.transpose(0,2,3,1)
        dout = dout.reshape(-1, FN)

        self.db = np.sum(dout)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0)
        self.dW = self.dW.reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        # 出ていくデータの高さ
        out_h = int((H + 2 * self.pad - FH) / self.stride + 1)
        # 出ていくデータの幅
        out_w = int((W + 2 * self.pad - FW) / self.stride + 1)
        dx = col2im(dcol, out_h, out_w, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

# プーリング層
class Pooling:
    def __init__(self, pool_h, pool_w, stride, pad):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        # 入ってくるデータのバッチ数、チャンネル、高さ、幅
        N, C, H, W = x.shape
        # 出ていくデータの高さ
        out_h = int(1 + (H - self.pool_h) / self.stride)
        # 出ていくデータの幅
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # im2colで展開
        col = im2col(x, out_h, out_w, self.pool_h, self.pool_w, self.stride, self.pad)
        # チャンネルごとに独立に展開
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        # 行ごと(1次元目の軸ごと)に最大値を求める
        out = np.max(col, axis=1)

        # 適切な出力サイズに整形する
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    # Reluの逆伝播参考
    def backward(self, dout):
        # 入ってくるデータのバッチ数、チャンネル、高さ、幅
        N, C, H, W = self.x.shape
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        # fllattenメソッドで平坦化
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

        # 出ていくデータの高さ
        out_h = int((H + 2 * self.pad - self.pool_h) / self.stride + 1)
        # 出ていくデータの幅
        out_w = int((W + 2 * self.pad - self.pool_w) / self.stride + 1)
        dx = col2im(dcol, out_h, out_w, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx