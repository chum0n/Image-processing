# image-processing

## ディレクトリ構成とその説明

### Keras/

python で書かれたニューラルネットワーク用の API である Keras を用いて画像認識プログラムを作成。(Keras はバックエンドとして TensorFlow や Theano を利用。)

※ 非公開

### LayerNet/

学習に使うニューラルネットワークの設計をまとめたディレクトリ。

### NeuralNetwork/

学習をする際に使うファイルをまとめたディレクトリ。ここで、使用するニューラルネットワークや最適化手法を設定。

### common/

学習に使う関数や層、最適化手法をまとめたディレクトリ。

### data/

学習する対象であるデータをまとめたディレクトリ。

### practice/

学習を始めるにあたって最初にデータの読み込みなどが正しくできているかを確認するためのファイルをまとめたディレクトリ。

### 全体図

```
.
├── Keras
│   ├── Keras_sample.py
│   ├── KerasforB11.py
│   ├── KerasforB12.py
│   └── KerasforB2.py
├── LayerNet
│   ├── LN.py
│   ├── LNforBN.py                      // Batch Normalizationを使用
│   ├── LNforCP.py                      // 畳み込み層・プーリング層を使用
│   ├── LNforConv.py                    // 畳み込み層を使用
│   ├── LNforDropout.py                 // Dropoutを使用
│   ├── LNforRelu.py                    // Relu関数を使用
│   └── SimpleConvNet.py
├── NeuralNetwork
│   ├── NNAdaDelta.py                   // 最適化手法にAdaDeltaを使用
│   ├── NNAdaGrad.py                    // 最適化手法にAdaGradを使用
│   ├── NNAdam.py                       // 最適化手法にAdamを使用
│   ├── NNMomentum.py                   // 最適化手法に慣性項付きSGDを使用
│   ├── NNRMSProp.py                    // 最適化手法にRMSPropを使用
│   ├── NNSGD.py                        // 最適化手法にSGDを使用
│   ├── NNSGDRelu.py                    // Relu関数を使用
│   ├── NNSGDforBN.py                   // Batch Normalizationを使用
│   ├── NNSGDforCP.py                   // 畳み込み層・プーリング層を使用
│   ├── NNSGDforConv.py  　　　　　　　　　// 畳み込み層を使用
│   ├── NNSGDforDropout.py              // Dropoutを使用
│   ├── NNcolor.py                      // cifarデータに対する学習
│   ├── NNtestforcolor.py               // cifarテストデータに対する正答率を求める
│   ├── NNtestformnist.py               // mnistテストデータに対する正答率を求める
│   ├── networkWh.npy                   // 学習にて得た重み
│   ├── networkWo.npy                   // 学習にて得た重み
│   ├── networkbh.npy                   // 学習にて得た重み
│   ├── networkbo.npy                   // 学習にて得た重み
│   ├── sample_color.py                 // cifar読み込み確認のためのサンプル
│   └── sample_mnist.py                 // mnist読み込み確認のためのサンプル
├── README.md
├── common
│   ├── functions.py                    // 種々の関数
│   ├── layers.py                       // 種々の層
│   ├── optimizer.py                    // 種々の最適化手法
│   └── util.py                         // 畳み込み層のための関数
├── data
│   ├── cifar-10-batches-py             // cifar画像データ
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   └── mnist                           // mnist画像データ
│       ├── t10k-images-idx3-ubyte
│       ├── t10k-labels-idx1-ubyte
│       ├── train-images-idx3-ubyte
│       └── train-labels-idx1-ubyte
└── practice                            
    ├── practice1.py
    └── practice2.py
```

参考文献はPerformance_evalution.pdf参照