from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from keras.layers import Input
from keras.models import Model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from keras.models import model_from_json

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential

dataset = pd.read_csv("Dataset/Modbus_dataset.csv")
dataset.fillna(0, inplace = True)
columns = dataset.columns
print(np.unique(dataset['label'], return_counts=True))
encode = []
for i in range(len(columns)):
    le = LabelEncoder()
    dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
    encode.append(le)
dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]
print(X)
print(X.shape)
print(Y)
print(np.unique(Y, return_counts=True))

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
'''
encoding_dim = 256 # encoding dimesnion is 32 which means each image will be filtered 32 times to get important features from images
input_size = keras.Input(shape=(X.shape[1],)) #we are taking input size
encoded = layers.Dense(encoding_dim, activation='relu')(input_size) #creating dense layer to start filtering dataset with given 32 filter dimension
decoded = layers.Dense(y_train.shape[1], activation='softmax')(encoded) #creating another layer with input size as 784 for encoding
autoencoder = keras.Model(input_size, decoded) #creating decoded layer to get prediction result
encoder = keras.Model(input_size, encoded)#creating encoder object with encoded and input images
encoded_input = keras.Input(shape=(encoding_dim,))#creating another layer for same input dimension
decoder_layer = autoencoder.layers[-1] #holding last layer
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))#merging last layer with encoded input layer
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#compiling model
print(autoencoder.summary())#printing model summary
hist = autoencoder.fit(X_train, y_train, epochs=30, batch_size=32, shuffle=True, validation_data=(X_test, y_test))#now start generating model with given Xtrain as input 
autoencoder.save_weights('model/encoder_model_weights.h5')#above line for creating model will take 100 iterations            
model_json = autoencoder.to_json() #saving model
with open("model/encoder_model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close    
f = open('model/encoder_history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()
encoder_acc = hist.history
acc = encoder_acc['accuracy']


with open('model/encoder_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    autoencoder = model_from_json(loaded_model_json)
json_file.close()
autoencoder.load_weights("model/encoder_model_weights.h5")
autoencoder._make_predict_function()
        
predict = autoencoder.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
f = f1_score(testY, predict,average='macro') * 100
a = accuracy_score(testY,predict)*100    
print(f)
print(a)
fpr, tpr, threshold = metrics.roc_curve(testY, predict, pos_label=1)
print((np.sum(fpr) / len(fpr)))
print((np.sum(tpr) / len(fpr)))
'''

'''
X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
print(X1.shape)
X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2)

cnn_model = Sequential()
cnn_model.add(Convolution2D(32, 1, 1, input_shape = (X1.shape[1], X1.shape[2], X1.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Convolution2D(32, 1, 1, activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim = 256, activation = 'relu'))
cnn_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
print(cnn_model.summary())
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = cnn_model.fit(X_train, y_train, batch_size=16, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test))
cnn_model.save_weights('model/cnn_model_weights.h5')            
model_json = cnn_model.to_json()
with open("model/cnn_model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()    
f = open('model/cnn_history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()

predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
f = f1_score(testY, predict,average='macro') * 100
a = accuracy_score(testY,predict)*100    
print(f)
print(a)
fpr, tpr, threshold = metrics.roc_curve(testY, predict, pos_label=1)
print((np.sum(fpr) / len(fpr)))
print((np.sum(tpr) / len(fpr)))
'''
'''
X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(X1.shape)
X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2)

lstm_model = Sequential()
lstm_model.add(keras.layers.LSTM(100,input_shape=(X1.shape[1], X1.shape[2])))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(100, activation='relu'))
lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
lstm_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(lstm_model.summary())
hist = lstm_model.fit(X_train, y_train, batch_size=16, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test))
lstm_model.save_weights('model/lstm_model_weights.h5')            
model_json = lstm_model.to_json()
with open("model/lstm_model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()    
f = open('model/lstm_history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()

predict = lstm_model.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
f = f1_score(testY, predict,average='macro') * 100
a = accuracy_score(testY,predict)*100    
print(f)
print(a)
fpr, tpr, threshold = metrics.roc_curve(testY, predict, pos_label=1)
print((np.sum(fpr) / len(fpr)))
print((np.sum(tpr) / len(fpr)))
'''

with open('model/cnn_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    autoencoder = model_from_json(loaded_model_json)
json_file.close()
autoencoder.load_weights("model/cnn_model_weights.h5")
autoencoder._make_predict_function()
labels = ['MITM_UNALTERED', 'NORMAL', 'RESPONSE_ATTACK']

test = pd.read_csv("Dataset/testData.csv")
test.fillna(0, inplace = True)
for i in range(len(encode)-1):
    test[columns[i]] = pd.Series(encode[i].transform(test[columns[i]].astype(str)))
test = test.values
test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
predict = autoencoder.predict(test)
predict = np.argmax(predict, axis=1)
print(predict)
    
