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

from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(rescale=1.0/255) 
test_datagen = ImageDataGenerator(rescale=1.0/255)

input_shape = (218, 178, 3) 
num_classes = 2
batch_size = 32

train_generator = train_datagen.flow_from_directory('/home/iiyama/face/train/',
    target_size=(218,178), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('/home/iiyama/face/test/',
    target_size=(218,178),
    batch_size=batch_size,
    class_mode='categorical')

# モデルの定義
model = models.Sequential()
# 3x3の畳み込み層.出力は32チャンネル.活性化関数にReLU.入力データのサイズは input_shape で指定.入出力のサイズ(row と col)が同じになるように設定. 
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu',
input_shape=input_shape, padding='same'))
# 3x3の畳み込み層.出力は64チャンネル.活性化関数にReLU.入力データのサイズは自動的に決まるので設定不要. 
model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
# 2x2の最大値プーリング 
model.add(layers.MaxPooling2D(pool_size=(2,2)))
# 入力を1次元配列に並び替える
model.add(layers.Flatten())
# 全結合層.出力ノード数は128.活性化関数にReLU. 
model.add(layers.Dense(128,activation='relu'))
# 全結合層.出力ノード数は128.活性化関数にReLU. 
model.add(layers.Dense(128,activation='relu'))
# 全結合層.出力ノード数はnum_classes(クラス数).活性化関数にsoftmax.
model.add(layers.Dense(num_classes, activation='softmax'))

# 作成したモデルの概要を表示
print (model.summary())

model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['acc'])
# 学習.
epochs = 10
spe = 100 # 1エポックあたりのバッチ数
result = model.fit_generator(train_generator, steps_per_epoch = spe,
    epochs=epochs, validation_data=test_generator, validation_steps=30)
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

loss, acc = model.evaluate_generator(test_generator, steps = 30)
print ("loss:", loss)
print ("accuracy:", acc)

# 損失関数と精度のグラフを表示.縦軸が損失関数の値(or精度)，横軸がエポック 数
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(history['loss'], label='loss') # 教師データの損失 
plt.plot(history['val_loss'], label='val_loss') # テストデータの損失 
plt.legend()
plt.savefig("loss_historyB2.png")

fig = plt.figure()
plt.plot(history['acc'], label='acc') # 教師データでの精度
plt.plot(history['val_acc'], label='val_acc') # テストデータでの精度
plt.legend()
plt.savefig("loss_accB2.png")