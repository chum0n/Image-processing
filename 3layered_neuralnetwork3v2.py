import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm
from LayerNetFor3 import LayerNet

INNODES = 784
HNODES = 100
ONODES = 10

network = LayerNet(INNODES, HNODES, ONODES)

ITER_NUM = 60000 # 勾配法による更新の回数