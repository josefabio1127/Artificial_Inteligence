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
data = pd.read_csv('/content/drive/MyDrive/IA/winequality-white.csv', sep = ';')

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
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                      'pH', 'sulphates', 'alcohol']].to_numpy() #entradas

#Step 4 - Creating the Training and Test datasets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.40)

#Step 5 - Define, compile, and fit the Keras classification model
np.random.seed(7)
tf.random.set_seed(7)

model = Sequential()
model.add(Input(shape=(11,))) #capa de entrada
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1)) #capa de salida

#Adjusting the learning rate
lr = LearningRateScheduler(lambda epoch: 0.0001)

# Compile the model
model.compile(optimizer='adam',
loss='mse',
metrics=['mae','mse'])

# build the model
mod = model.fit(X_train, Y_train, epochs = 20, validation_data = (X_test, Y_test),
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
plt.ylim([0,2])
plt.legend()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [$MPG^2$]')
plt.plot(hist['epoch'], hist['mse'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mse'],
         label = 'Val Error')
plt.ylim([0,1])
plt.legend()
plt.show()

#Step 6 - Predictions
data_prueba = [[5.9, 0.66, 0.58, 2.3, 77, 52, 12, 0.9733, 4.2, 0.56, 10.2],
               [8.7, 0.16, 0.26, 9.3, 25, 16, 34, 0.9933, 3.2, 0.65, 12.2],
               [7.2, 0.64, 0.42, 6.3, 56, 15, 22, 0.9018, 1.2, 1.08, 8.2]]
data_pandas = pd.DataFrame(data_prueba, columns = ['fixed acidity', 'volatile acidity',
                                                   'citric acid', 'residual sugar', 'chlorides',
                                                   'free sulfur dioxide', 'total sulfur dioxide',
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
