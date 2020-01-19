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