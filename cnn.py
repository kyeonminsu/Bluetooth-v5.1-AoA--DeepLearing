import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

np.random.seed(1337)

def windows(data,size):
    start = 0
    while start< len(test_data):
        yield int(start), int(start + size)
        start+= (size/10)

def segment_signal(data, window_size = 10):
    segments = np.empty((0, window_size, 2))
    labels= np.empty((0))
    for (start, end) in windows(data['ant1'],window_size):
        x = data['ant1'][start:end]
        print(start)
        print(end)
        y = data['ant2'][start:end]
        if(len(data['ant1'][start:end])==window_size):
            segments = np.vstack([segments,np.dstack([x,y])])
            labels = np.append(labels,stats.mode(data['table'][start:end])[0][0])
    return segments, labels


print("0")

c =['ant1', 'ant2', 'table']
test_data = pd.read_csv('Train.csv',header = None, names=c)
x, y = segment_signal(test_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, )

enc=OneHotEncoder(categories='auto', sparse=True)
train_test_y = y_train.reshape(-1,1)
enc.fit(train_test_y)
train_test_y_onehot = enc.transform(train_test_y).toarray()

print("1")

test_test_y = y_test.reshape(-1,1)
enc.fit(test_test_y)
test_test_y_onehot = enc.transform(test_test_y).toarray()
train_x_1 = x_train.reshape(-1,10,1,2)
test_x_1 = x_test.reshape(-1,10,1,2)

print("2")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 6), padding='same', input_shape=(10, 1, 2), activation='relu',
                           kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
    tf.keras.layers.Conv2D(32, (3, 6), padding='same', activation='relu',
                           kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
    tf.keras.layers.Conv2D(64, (3, 6), padding='same', activation='relu',
                           kernel_initializer=tf.keras.initializers.he_normal(seed=None)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
    tf.keras.layers.Dense(13, activation='softmax')
])

model1.summary()
lr = 0.01
model1.compile(optimizer=tf.keras.optimizers.Adam(),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(patience = 3)
batch_size = 10
epochs = 20
model1.fit(train_x_1, train_test_y_onehot,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_data=(test_x_1, test_test_y_onehot),
           callbacks=[early_stop]
          )

lim = model1.predict(train_x_1,batch_size=10)

model1.save('model_aoa.h5')

converter = tf.lite.TFLiteConverter.from_keras_model_file('model.h5')
tfmodel = converter.convert()
open("model_aoa.tflite","wb").write(tfmodel)