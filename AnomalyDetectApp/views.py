from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pymysql
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
import matplotlib.pyplot as plt

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import seaborn as sns

global X, Y, dataset, encode, X_train, X_test, y_train, y_test
global algorithms, accuracy, f1, tpr, fpr, cnn_model, columns
labels = ['MITM_UNALTERED', 'NORMAL', 'RESPONSE_ATTACK']


def ProcessData(request):
    if request.method == 'GET':
        global X, Y, dataset, encode, columns, X_train, X_test, y_train, y_test
        encode = []
        dataset = pd.read_csv("Dataset/Modbus_dataset.csv")
        dataset.fillna(0, inplace = True)
        temp = dataset.values
        columns = dataset.columns
        label = dataset.groupby('label').size()
        encode = []
        for i in range(len(columns)):
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
            encode.append(le)
        dataset = dataset.values
        X = dataset[:,0:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(columns)):
            output += "<th>"+font+columns[i]+"</th>"
        output += "</tr>"
        for i in range(0,100):
            output += "<tr>"
            for j in range(0,temp.shape[1]):
                output += '<td><font size="" color="black">'+str(temp[i,j])+'</td>'
            output += "</tr>"    
        context= {'data': output}
        label.plot(kind="bar")
        plt.title("Various Attacks found in Modbus/TCP dataset")
        plt.show()
        return render(request, 'UserScreen.html', context)
        

def TrainPropose(request):
    global X, Y, dataset, encode, X_train, X_test, y_train, y_test, labels
    global algorithms, accuracy, f1, tpr, fpr, cnn_model, columns
    algorithms = []
    accuracy = []
    f1 = []
    tpr = []
    fpr = []
    if request.method == 'GET':
        if os.path.exists("model/encoder_model.json"):
            with open('model/encoder_model.json', "r") as json_file:
                loaded_model_json = json_file.read()
                autoencoder = model_from_json(loaded_model_json)
            json_file.close()
            autoencoder.load_weights("model/encoder_model_weights.h5")
            autoencoder.make_predict_function()
        else:
            encoding_dim = 256 # encoding dimesnion is 32 which means each image will be filtered 32 times to get important features from images
            input_size = keras.Input(shape=(X.shape[1],)) #we are taking input size
            encoded = layers.Dense(encoding_dim, activation='relu')(input_size) #creating dense layer to start filtering dataset with given 32 filter dimension
            decoded = layers.Dense(y_train.shape[1], activation='softmax')(encoded) #creating another layer with input size as dataset for encoding
            autoencoder = keras.Model(input_size, decoded) #creating decoded layer to get prediction result
            encoder = keras.Model(input_size, encoded)#creating encoder object with encoded and input images
            encoded_gan = keras.Input(shape=(encoding_dim,))#creating another layer for same input dimension with GAN
            decoder_layer = autoencoder.layers[-1] #holding last layer
            decoder = keras.Model(encoded_input, decoder_layer(encoded_gan))#merging last layer with encoded input layer
            autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#compiling model
            hist = autoencoder.fit(X_train, y_train, epochs=30, batch_size=32, shuffle=True, validation_data=(X_test, y_test))#now start generating model with given Xtrain as input 
            autoencoder.save_weights('model/encoder_model_weights.h5')#above line for creating model will take 100 iterations            
            model_json = autoencoder.to_json() #saving model
            with open("model/encoder_model.json", "w") as json_file:
                json_file.write(model_json)
            json_file.close    
            f = open('model/encoder_history.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()
        print(autoencoder.summary())#printing model summary
        predict = autoencoder.predict(X_test)
        predict = np.argmax(predict, axis=1)
        testY = np.argmax(y_test, axis=1)
        f = f1_score(testY, predict,average='macro')
        a = accuracy_score(testY,predict)    
        fprs, tprs, threshold = metrics.roc_curve(testY, predict, pos_label=1)
        fpr_value = (np.sum(fprs) / len(fprs))
        tpr_value = (np.sum(tprs) / len(fprs))
        accuracy.append(a)
        f1.append(f)
        tpr.append(tpr_value)
        fpr.append(fpr_value)
        algorithms.append("Propose AutoEncoder-GAN MENSA")
        arr = ['Algorithm Name', 'Accuracy', 'TPR', 'FPR', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(tpr[i])+"</td><td>"+font+str(fpr[i])+"</td><td>"+font+str(f1[i])+"</td></tr>"
        context= {'data': output}
        conf_matrix = confusion_matrix(testY, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,len(labels)])
        plt.title("Propose AutoEncoder-GAN MENSA Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
        return render(request, 'UserScreen.html', context)

def TrainCNN(request):
    if request.method == 'GET':
        global X, Y, dataset, encode, X_train, X_test, y_train, y_test, labels
        global algorithms, accuracy, f1, tpr, fpr, cnn_model, columns
        X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
        print(X1.shape)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2)
        if request.method == 'GET':
            if os.path.exists("model/cnn_model.json"):
                with open('model/cnn_model.json', "r") as json_file:
                    loaded_model_json = json_file.read()
                    cnn_model = model_from_json(loaded_model_json)
                json_file.close()
                cnn_model.load_weights("model/cnn_model_weights.h5")
                #cnn_model._make_predict_function()
            else:
                cnn_model = Sequential()
                #defining convolution 2D neural network with 32 filters
                cnn_model.add(Convolution2D(32, 1, 1, input_shape = (X1.shape[1], X1.shape[2], X1.shape[3]), activation = 'relu'))
                #defining max pooling layer to collect filtered data
                cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
                cnn_model.add(Convolution2D(32, 1, 1, activation = 'relu'))
                cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
                cnn_model.add(Flatten())
                cnn_model.add(Dense(output_dim = 256, activation = 'relu'))
                cnn_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
                cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compiling model
                #start training model
                hist = cnn_model.fit(X_train, y_train, batch_size=16, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test))
                cnn_model.save_weights('model/cnn_model_weights.h5')            
                model_json = cnn_model.to_json()
                with open("model/cnn_model.json", "w") as json_file:
                    json_file.write(model_json)
                json_file.close()    
                f = open('model/cnn_history.pckl', 'wb')
                pickle.dump(hist.history, f)
                f.close()            
        print(cnn_model.summary())#printing model summary
        predict = cnn_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        testY = np.argmax(y_test, axis=1)
        f = f1_score(testY, predict,average='macro')
        a = accuracy_score(testY,predict)    
        fprs, tprs, threshold = metrics.roc_curve(testY, predict, pos_label=1)
        fpr_value = (np.sum(fprs) / len(fprs))
        tpr_value = (np.sum(tprs) / len(fprs))
        accuracy.append(a)
        f1.append(f)
        tpr.append(tpr_value)
        fpr.append(fpr_value)
        algorithms.append("CNN Algorithm")
        arr = ['Algorithm Name', 'Accuracy', 'TPR', 'FPR', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(tpr[i])+"</td><td>"+font+str(fpr[i])+"</td><td>"+font+str(f1[i])+"</td></tr>"
        context= {'data': output}
        conf_matrix = confusion_matrix(testY, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,len(labels)])
        plt.title("CNN Algorithm Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()    
        return render(request, 'UserScreen.html', context)

def TrainLSTM(request):
    if request.method == 'GET':
        global X, Y, dataset, encode, X_train, X_test, y_train, y_test, labels
        global algorithms, accuracy, f1, tpr, fpr, columns
        X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
        print(X1.shape)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2)
        if request.method == 'GET':
            if os.path.exists("model/lstm_model.json"):
                with open('model/lstm_model.json', "r") as json_file:
                    loaded_model_json = json_file.read()
                    lstm_model = model_from_json(loaded_model_json)
                json_file.close()
                lstm_model.load_weights("model/lstm_model_weights.h5")
                #lstm_model._make_predict_function()
            else:
                lstm_model = Sequential()#creating sequenctial object
                #adding LSTM layer to sequential object with 100 filters
                lstm_model.add(keras.layers.LSTM(100,input_shape=(X1.shape[1], X1.shape[2])))
                lstm_model.add(Dropout(0.5))
                lstm_model.add(Dense(100, activation='relu'))
                lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
                #compiling the model
                lstm_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
                #start training LSTM model
                hist = lstm_model.fit(X_train, y_train, batch_size=16, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test))
                lstm_model.save_weights('model/lstm_model_weights.h5')            
                model_json = lstm_model.to_json()
                with open("model/lstm_model.json", "w") as json_file:
                    json_file.write(model_json)
                json_file.close()    
                f = open('model/lstm_history.pckl', 'wb')
                pickle.dump(hist.history, f)
                f.close()                 
        print(lstm_model.summary())#printing model summary
        predict = lstm_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        testY = np.argmax(y_test, axis=1)
        f = f1_score(testY, predict,average='macro')
        a = accuracy_score(testY,predict)    
        fprs, tprs, threshold = metrics.roc_curve(testY, predict, pos_label=1)
        fpr_value = (np.sum(fprs) / len(fprs))
        tpr_value = (np.sum(tprs) / len(fprs))
        accuracy.append(a)
        f1.append(f)
        tpr.append(tpr_value)
        fpr.append(fpr_value)
        algorithms.append("LSTM Algorithm")
        arr = ['Algorithm Name', 'Accuracy', 'TPR', 'FPR', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(tpr[i])+"</td><td>"+font+str(fpr[i])+"</td><td>"+font+str(f1[i])+"</td></tr>"
        context= {'data': output}
        conf_matrix = confusion_matrix(testY, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,len(labels)])
        plt.title("LSTM Algorithm Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()    
        return render(request, 'UserScreen.html', context)

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global cnn_model, encode, columns, labels
        testFile = request.POST.get('t1', False)
        test = pd.read_csv("Dataset/testData.csv")
        test.fillna(0, inplace = True)
        temp = test.values
        for i in range(len(encode)-1):
            test[columns[i]] = pd.Series(encode[i].transform(test[columns[i]].astype(str)))
        test = test.values
        test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
        predict = cnn_model.predict(test)
        predict = np.argmax(predict, axis=1)
        arr = ['Test Data', 'Detection & Classification Result']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(predict)):
            output +="<tr><td>"+font+str(temp[i])+"</td><td>"+font+str(labels[predict[i]])+"</td></tr>"
        context= {'data': output}    
        return render(request, 'UserScreen.html', context) 


def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})  

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})


def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'AnomalyDetect',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+uname}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed. Please retry'}
            return render(request, 'UserLogin.html', context)        

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'AnomalyDetect',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'AnomalyDetect',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,gender,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+gender+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Completed'
        context= {'data':output}
        return render(request, 'Signup.html', context)
      


