# image-processing

## ディレクトリ構成とその説明

ディレクトリごとの意味について簡単に説明する

### Keras ディレクトリ

python で書かれたニューラルネットワーク用の API である Keras を用いて画像認識プログラムを作成する。Keras はバックエンドとして TensorFlow や Theano を利用する。

### LayerNet ディレクトリ

学習に使うニューラルネットワークの設計をまとめたディレクトリである。

### NeuralNetwork ディレクトリ

学習をする際に使うファイルをまとめたディレクトリである。ここで使用するニューラルネットワークや最適化手法を設定する。

### common ディレクトリ

学習に使う関数や層、最適化手法をまとめたディレクトリである。

### data ディレクトリ

学習する対象であるデータをまとめたディレクトリである。

### practice ディレクトリ

学習を始めるにあたって最初にデータの読み込みなどが正しくできているかを確認するためのファイルをまとめたディレクトリである。

### 全体図

以下がディレクトリ構成全体である。
```
.
├── Keras
│   ├── Keras_sample.py
│   ├── KerasforB11.py
│   ├── KerasforB12.py
│   └── KerasforB2.py
├── LayerNet
│   ├── LN.py
│   ├── LNforBN.py
│   ├── LNforCP.py
│   ├── LNforConv.py
│   ├── LNforDropout.py
│   ├── LNforRelu.py
│   └── SimpleConvNet.py
├── NeuralNetwork
│   ├── NNAdaDelta.py
│   ├── NNAdaGrad.py
│   ├── NNAdam.py
│   ├── NNMomentum.py
│   ├── NNRMSProp.py
│   ├── NNSGD.py
│   ├── NNSGDRelu.py
│   ├── NNSGDforBN.py
│   ├── NNSGDforCP.py
│   ├── NNSGDforConv.py
│   ├── NNSGDforDropout.py
│   ├── NNcolor.py
│   ├── NNtestforcolor.py
│   ├── NNtestformnist.py
│   ├── networkWh.npy
│   ├── networkWo.npy
│   ├── networkbh.npy
│   ├── networkbo.npy
│   ├── sample_color.py
│   └── sample_mnist.py
├── README.md
├── common
│   ├── functions.py
│   ├── layers.py
│   ├── optimizer.py
│   └── util.py
├── data
│   ├── cifar-10-batches-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   └── mnist
│       ├── t10k-images-idx3-ubyte
│       ├── t10k-labels-idx1-ubyte
│       ├── train-images-idx3-ubyte
│       └── train-labels-idx1-ubyte
└── practice
    ├── practice1.py
    └── practice2.py
```