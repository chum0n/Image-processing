import numpy as np

def identity_function(x): # 恒等関数
    return x

# シグモイド関数(活性化関数)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# # ソフトマックス関数
# def softmax(a):
#     # 一番大きい値を取得
#     c = np.max(a)
#     # 各要素から一番大きな値を引く（オーバーフロー対策）
#     exp_a = np.exp(a - c)
#     sum_exp_a = np.sum(exp_a)
#     # 要素の値/全体の要素の合計
#     y = exp_a / sum_exp_a

#     return y

# # ソフトマックス関数
# def softmax(a):
#     # 一番大きい値を取得
#     c = np.max(a, axis=1, keepdims=True)
#     # 各要素から一番大きな値を引く（オーバーフロー対策）
#     exp_a = np.exp(a - c)
#     sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
#     # 要素の値/全体の要素の合計
#     y = exp_a / sum_exp_a

#     return y

# ソフトマックス関数
def softmax(x):
    if x.ndim == 2: # バッチで二次元の時
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

# バッチ対応版クロスエントロピー誤差
def cross_entropy_error(y, t):
    # 一枚に対してのとき
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    # if t.size == y.size:
    #     t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return -np.sum(t * np.log(y + 1e-7)) / batch_size