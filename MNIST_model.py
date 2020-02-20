import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import keras
import matplotlib.pyplot as plt
import cv2
from keras.optimizers import SGD
from PIL import Image
from scipy.misc import imread,imresize
from sklearn.externals import joblib

df1 = pd.read_csv('train.csv')
X_train = np.array(df1.iloc[:,1:])
y_train = np.array(df1.iloc[:,:1])

df2 = pd.read_csv('test.csv')
x_test = df1.iloc[:,1:]

num_classes = 10
y_train = keras.utils.to_categorical(y_train,num_classes)

img_size = 784

model = Sequential()
model.add(Dense(units=32,activation='sigmoid',input_shape=(img_size,)))
#model.add(Dense(units=16,activation='relu',input_shape=(32,)))
model.add(Dense(units=num_classes,activation='softmax'))
model.summary()

opt = SGD(lr=0.01)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=200,epochs=50,validation_split=0.1)

joblib.dump(model,'mnist_model.pkl')


im = x_test.iloc[800:801,0:]
plt.imshow(im.values.reshape((28,28)))
model_from_pkl = joblib.load('mnist_model.pkl')
#print(im.shape)
y_hat = model_from_pkl.predict(im)
y_pred = model.predict(im)
#print(y_pred)
np.argmax(y_pred)



x = imread('test15.png', mode='L')
print(x.shape)
plt.imshow(x.reshape((28,28)))
test_pred = model.predict(x.reshape((1, 784)))
print(test_pred)
np.argmax(test_pred)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['training','validation'],loc='best')
plt.show()
