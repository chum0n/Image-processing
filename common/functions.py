import numpy as np

# 恒等関数
def identity_function(x):
    return x

# シグモイド関数(活性化関数)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ソフトマックス関数
def softmax(x):
    # バッチで二次元の時
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    # オーバーフロー対策
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# バッチ対応版クロスエントロピー誤差
def cross_entropy_error(y, t):
    # 一枚に対してのとき
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size