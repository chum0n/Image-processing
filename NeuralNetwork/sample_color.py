import numpy as np
import pickle
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    X =np.array(dict[b'data'])
    X =X.reshape((X.shape[0],3,32,32))
    Y =np.array(dict[b'labels'])
    return X,Y

X,Y = unpickle("../data/cifar-10-batches-py/data_batch_1")

import matplotlib.pyplot as plt
idx = 1000
plt.imshow(X[idx].transpose(((1,2,0)))) # X[idx] が (3*32*32) になっているのを (32*32*3) に変更する.

plt.show() # トラックの画像が表示されるはず

print(Y[idx]) # 9番(truck)が表示されるはず