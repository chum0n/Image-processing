import numpy as np

# 確率的勾配降下法
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# 慣性項付きSGD(Momentum)
class Momentum:
    def __init__(self, lr=0.01, alfa=0.9):
        self.lr = lr
        self.alfa = alfa
        self.deltaW = None

    def update(self, params, grads):
        # updateが初めて呼ばれるとき
        # パラメータと同じ構造のデータをディクショナリ変数として保持
        if self.deltaW is None:
            self.deltaW = {}
            for key, val in params.items():
                self.deltaW[key] = np.zeros_like(val)

        for key in params.keys():
            self.deltaW[key] = self.alfa * self.deltaW[key] - self.lr * grads[key]
            params[key] += self.deltaW[key]

# AdaGrad
class AdaGrad:
    def __init__(self, lr=0.001):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr / (np.sqrt(self.h[key]) + 1e-8) * grads[key] # 0で割ってしまうことを防ぐ

# RMSProp
class RMSprop:
    def __init__(self, lr=0.001, rho = 0.9):
        self.lr = lr
        self.rho = rho
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.rho
            self.h[key] += (1 - self.rho) * grads[key] * grads[key]
            params[key] -= self.lr / (np.sqrt(self.h[key]) + 1e-8) * grads[key]

# AdaDelta
class AdaDelta:
    def __init__(self, rho = 0.95):
        self.rho = rho
        self.h = None
        self.s = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            self.s = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                self.s[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.rho
            self.h[key] += (1 - self.rho) * grads[key] * grads[key]
            deltaW = -(np.sqrt(self.s[key] + 1e-6)) / (np.sqrt(self.h[key] + 1e-6)) * grads[key]
            self.s[key] *= self.rho
            self.s[key] += (1 - self.rho) * deltaW * deltaW
            params[key] += deltaW

# Adam
class Adam:
    def __init__(self, alfa=0.001, beta1=0.9, beta2=0.999):
        self.alfa = alfa
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = None
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.t is None:
            self.t = 0
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        for key in params.keys():
            self.m[key] *= self.beta1
            self.m[key] += (1 - self.beta1) * grads[key]
            self.v[key] *= self.beta2
            self.v[key] += (1 - self.beta2) * grads[key] * grads[key]
            m2 = self.m[key] / (1 - np.power(self.beta1, self.t))
            v2 = self.v[key] / (1 - np.power(self.beta2, self.t))
            params[key] -= self.alfa * m2 / (np.sqrt(v2) + 1e-8)