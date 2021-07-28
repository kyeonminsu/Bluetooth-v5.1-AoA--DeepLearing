
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling1D, Conv1D, LSTM, TimeDistributed
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()
np.random.seed(1337)


c =['ant1', 'ant2', 'table'] 
test_data = pd.read_csv('clean_dirty_sum_add_front.csv',header = None, names=c)


def windows(data,size):
    start = 0
    while start< len(test_data):
        yield int(start), int(start + size)
        start+= (size/5)



def segment_signal(data, window_size = 10):
    segments = np.empty((0, window_size, 2))
    labels= np.empty((0))
    for (start, end) in windows(data['ant1'],window_size):
        x = data['ant1'][start:end]
        y = data['ant2'][start:end]
        if(len(data['ant1'][start:end])==window_size):
            segments = np.vstack([segments,np.dstack([x,y])])
            labels = np.append(labels,stats.mode(data['table'][start:end])[0][0])
    return segments, labels


x, y = segment_signal(test_data)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, )

enc=OneHotEncoder(categories='auto', sparse=True)

train_test_y = y_train.reshape(-1,1)  
enc.fit(train_test_y)
train_test_y_onehot = enc.transform(train_test_y).toarray()

test_test_y = y_test.reshape(-1,1)  
enc.fit(test_test_y)
test_test_y_onehot = enc.transform(test_test_y).toarray()


train_x_1 = x_train.reshape(-1,10,1,2)
test_x_1 = x_test.reshape(-1,10,1,2)



model1=tf.keras.Sequential()
model1.add(Conv2D(16, (2,6),padding='same',input_shape=(10,1,2),activation='relu',kernel_initializer=keras.initializers.he_normal(seed=None)))
model1.add(Conv2D(32, (2,6),padding='same',activation='relu',kernel_initializer=keras.initializers.he_normal(seed=None)))
model1.add(Conv2D(64, (2,6),padding='same',activation='relu',kernel_initializer=keras.initializers.he_normal(seed=None)))
model1.add(Flatten())
model1.add(Dense(100, activation='relu',kernel_initializer=keras.initializers.he_normal(seed=None)))
model1.add(Dense(100, activation='relu',kernel_initializer=keras.initializers.he_normal(seed=None)))
model1.add(Dense(100, activation='relu',kernel_initializer=keras.initializers.he_normal(seed=None)))
model1.add(Dense(13,activation='softmax'))

model1.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model1.summary()



early_stop = EarlyStopping(patience = 5)





batch_size = 10
epochs = 5
model1.fit(train_x_1, train_test_y_onehot,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_data=(test_x_1, test_test_y_onehot),
           callbacks=[early_stop]
          )





c2 =['ant1', 'ant2', 'table'] 
pre_data = pd.read_csv('Pred_sum.csv',header = None, names=c2)




pre_x, pre_y = segment_signal(pre_data)
pre_x = pre_x.reshape(-1,10,1,2)

pre_y_one = pre_y.reshape(-1,1)  
enc.fit(pre_y_one)
pre_y_one = enc.transform(pre_y_one).toarray()



print(pre_x[2])
print(pre_y[2])
#print(len(pre_y))




lim = model1.predict(pre_x,batch_size=10,verbose=2) #_classes

#print(lim[100])
#print(pre_y[0])

#print(pre_y_one[0])
classes = [-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]
#for i in range(0,3000):
    #print(np.round(pre_y_one[i]),1)
    #print(np.round(lim[i]))

#print(lim.shape)
#print(pre_y_one.shape)
#print(tf.argmax(pre_y_one, 1)[100])
#print(tf.argmax(lim, 1)[100])

#con_mat=tf.math.confusion_matrix(labels=tf.argmax(pre_y_one, 1), predictions=tf.argmax(lim, 1)).numpy()
con_mat=confusion_matrix(tf.argmax(pre_y_one, 1),tf.argmax(lim, 1))

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,index = classes, columns = classes)


model1.save('model_aoa.h5')





converter = tf.lite.TFLiteConverter.from_keras_model_file('model_aoa.h5')
tfmodel = converter.convert()
open("CNN_model.tflite","wb").write(tfmodel)





figure = plt.figure(figsize=(7, 7))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()






