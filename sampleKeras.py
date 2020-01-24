import numpy as np
# Keras関係のimport
import keras
from keras import datasets, models, layers

# GPGPUリソースを全消費してしてしまう問題の回避
import tensorflow as tf
from keras.backend import set_session, tensorflow_backend
## 動的に必要なメモリだけを確保(allow_growth=True)
## デバイス「0」だけを利用 (visible_device_list="0") ※"0"の部分は，"0"~"3"の中から空いているものを選んでください
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,
  visible_device_list="3"))
set_session(tf.Session(config=config))

# MNISTデータの準備
img_rows, img_cols = 28, 28 # 画像サイズは 28x28 
num_classes = 10 # クラス数

(X, Y), (Xtest, Ytest) = keras.datasets.mnist.load_data() # 訓 練 用 と テスト(兼 validation)用のデータを取得
X = X.reshape(X.shape[0],img_rows,img_cols,1) # X を (画 像 ID， 28, 28, 1) の4次元配列に変換
Xtest = Xtest.reshape(Xtest.shape[0],img_rows,img_cols,1)

X = X.astype('float32') / 255.0 # 各画素の値を 0~1 に正規化 
Xtest = Xtest.astype('float32') /255.0

input_shape = (img_rows, img_cols, 1)

Y = keras.utils.to_categorical(Y, num_classes) # one-hot-vectorへ変換 
Ytest1 = keras.utils.to_categorical(Ytest, num_classes)

# モデルの定義
model = models.Sequential()
# 3x3の畳み込み層.出力は32チャンネル.活性化関数にReLU.入力データのサイ ズは input_shape で指定.入出力のサイズ(row と col)が同じになるように設定. 
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu',
input_shape=input_shape, padding='same'))
# 3x3の畳み込み層.出力は64チャンネル.活性化関数にReLU.入力データのサイ ズは自動的に決まるので設定不要. 
model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
# 2x2の最大値プーリング 
model.add(layers.MaxPooling2D(pool_size=(2,2)))
# 入力を1次元配列に並び替える
model.add(layers.Flatten())
# 全結合層.出力ノード数は128.活性化関数にReLU. 
model.add(layers.Dense(128,activation='relu'))
# 全結合層.出力ノード数はnum_classes(クラス数).活性化関数にsoftmax.
model.add(layers.Dense(num_classes, activation='softmax'))

# 作成したモデルの概要を表示
print (model.summary())

# モデルのコンパイル.損失関数や最適化手法を指定する.
# ここでは，損失関数にクロスエントロピー誤差，最適化手法にAdadeltaを指定し ている.
model.compile(
loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['acc'])
# 学習.とりあえずここでは 10 エポックだけ学習. 1バッチあたり 32 枚の画像を 利用
epochs = 12
batch_size = 32
result = model.fit(X,Y, batch_size=batch_size,
epochs=epochs, validation_data=(Xtest,Ytest1))
history = result.history

# 学習済みのモデルをファイルに保存したい場合
model.save("my_model.h5")

# ファイルに保存したモデルを読み込みたい場合
model = models.load_model("my_model.h5")

# 学習履歴をファイルに保存したい場合
import pickle
with open("my_model_history.dump", "wb") as f:
    pickle.dump(history, f)

# 学習履歴をファイルから読み込みたい場合
#with open("my_model_history.dump", "rb") as f:
# history = pickle.load(f)

# Xtestに対してクラス識別.
pred = model.predict_classes(Xtest)

from sklearn.metrics import confusion_matrix, accuracy_score
# 混同行列 各行が正解のクラス，各列が推定されたクラスに対応
print (confusion_matrix(Ytest, pred, labels=np.arange(10)))
# 正答率の算出
print (accuracy_score(Ytest, pred))

# 損失関数と精度のグラフを表示.縦軸が損失関数の値(or精度)，横軸がエポック 数
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(history['loss'], label='loss') # 教師データの損失 plt.plot(history[’val_loss’], label=’val_loss’) # テストデータの損失 plt.legend()
plt.savefig("loss_history.png")

fig = plt.figure()
plt.plot(history['acc'], label='acc') # 教師データでの精度
plt.plot(history['val_acc'], label='val_acc') # テストデータでの精度
plt.legend()
plt.savefig("loss_acc.png")