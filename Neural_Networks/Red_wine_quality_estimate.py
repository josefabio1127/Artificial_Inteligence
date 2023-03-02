#Step 1 - Loading the required libraries and modules

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import Input
from keras.callbacks import LearningRateScheduler

#Step 2 - Reading the Data
data = pd.read_csv('/content/drive/MyDrive/IA/winequality-red.csv', sep = ';')

#Step 3 - Creating arrays for the features and the response variable
def normalize(data):
    for column in data:
        if data[column].name == 'quality':
            data[column] = data[column]/10
            break
        else:
            data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    return data

data_norm = normalize(data)

labels = data_norm[['quality']].to_numpy() #salidas
features = data_norm[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                      'alcohol']].to_numpy() #entradas

#Step 4 - Creating the Training and Test datasetsX_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.40)
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.40)

#Step 5 - Define, compile, and fit the Keras classification model
np.random.seed(7)
tf.random.set_seed(7)

model = Sequential()
model.add(Input(shape=(11,))) #capa de entrada
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1)) #capa de salida

#Adjusting the learning rate
lr = LearningRateScheduler(lambda epoch: 1e-3)

# Compile the model
model.compile(optimizer='Adam',
              loss='mse',
              metrics=['mae','mse'])

# build the model
mod = model.fit(X_train, Y_train, epochs = 10,
                validation_data = (X_test, Y_test),
                callbacks= [lr])

model.summary()
print(mod.history)
hist = pd.DataFrame(mod.history)
hist['epoch'] = mod.epoch

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [MPG]')
plt.plot(hist['epoch'], hist['mae'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mae'],
         label = 'Val Error')
plt.ylim([0,1])
plt.legend()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [$MPG^2$]')
plt.plot(hist['epoch'], hist['mse'],
         label='Train Error')
plt.plot(hist['epoch'],hist['val_mse'],
         label = 'Val Error')
plt.ylim([0,1])
plt.legend()
plt.show()

#Step 6 - Predictions
data_prueba = [[6.3, 0.56, 0.18, 5.3, 177, 17, 23, 0.8933, 8.2, 0.91, 11.7],
               [9.9, 0.26, 0.02, 6.9, 24, 17, 51, 0.8733, 2.3, 0.67, 9.2],
               [5.9, 0.77, 0.12, 7.2, 112, 9, 67, 0.8841, 6.5, 1.17, 14.5]]

data_pandas = pd.DataFrame(data_prueba, columns = ['fixed acidity', 'volatile acidity',
                                                   'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                                                   'density', 'pH', 'sulphates', 'alcohol'])
total_data = pd.concat([data,data_pandas])
data_prueba_norm = normalize(total_data)
data_norm = data_prueba_norm.tail(3)
dat = data_norm[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']].to_numpy()
print("Datos")
print(dat)
print("Resultados")

results = model.predict(dat)
print(results*10)
