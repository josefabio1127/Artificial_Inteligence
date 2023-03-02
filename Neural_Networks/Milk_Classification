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
data = pd.read_csv('/content/drive/MyDrive/IA/milknew.csv')

#Step 3 - Creating arrays for the features and the response variable

#normalizacion de datos y adaptacion de la columna grade a las salidas de la red
def normalize(data):
    #arrays para las 3 neuronas de salida
    high = []
    medium = []
    low = []
    for column in data:
        if data[column].name == "pH":
            data[column] = (data[column] - 3) / 6.5 #normalización
            continue
        elif data[column].name == "Temprature":
            data[column] = (data[column] - 22) / 68 #normalización
            continue
        elif data[column].name == "Taste":
            continue
        elif data[column].name == "Odor":
            continue
        elif data[column].name == "Fat ":
            continue
        elif data[column].name == "Turbidity":
            continue
        elif data[column].name == "Colour":
            data[column] = (data[column] - 208) / 49 #normalización
            continue
        elif data[column].name == "Grade":
            for element in data[column]:
                if element == 'high':
                    high.append(1)
                    medium.append(0)
                    low.append(0)
                elif element == 'medium':
                    high.append(0)
                    medium.append(1)
                    low.append(0)
                else: #caso low
                    high.append(0)
                    medium.append(0)
                    low.append(1)
        del data['Grade']
        data['High'] = high
        data['Medium'] = medium
        data['Low'] = low
        break
    return data
    
#datos normalizados
norm_data = normalize(data)

#division de entradas y salidas
features = norm_data[['pH', 'Temprature', 'Taste', 'Odor', 'Fat ', 'Turbidity', 'Colour']].t
o_numpy() #entradas
labels = norm_data[['High', 'Medium', 'Low']].to_numpy() #salidas

#Step 4 - Creating the Training and Test datasets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.40)

#Step 5 - Define, compile, and fit the Keras classification model
np.random.seed(7)
tf.random.set_seed(7)

#Definicion del modelo
model = Sequential()
model.add(Input(shape=(7,))) #capa de entrada
model.add(Dense(9, activation='sigmoid'))
model.add(Dense(3)) #capa de salida

#Ajuste de la taza de aprendizaje
lr = LearningRateScheduler(lambda epoch: 0.0001)

# Compilación del modelo
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['binary_accuracy', 'categorical_crossentropy'])
              
# Construcción del modelo
mod = model.fit(X_train, Y_train, epochs = 100, validation_data = (X_test, Y_test), callback
s = [lr])

# Resumen de la red
model.summary()

# Obtención del historial de las métricas de la red
hist = pd.DataFrame(mod.history)
hist['epoch'] = mod.epoch

#Construcción de gráficos de error

#Gráfico 1: interaciones vs Perdida (de train y de test)
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.plot(hist['epoch'], hist['categorical_crossentropy'],
          label='Train Error')
plt.plot(hist['epoch'], hist['val_categorical_crossentropy'],
          label = 'Val Error')
plt.ylim([0,2.5]) #rango en y
plt.legend()

#Mostrar gráficos
plt.show()

#Step 6 - Predictions
data_prueba = [[7.5, 68, 1, 0, 1, 1, 238], [5.1, 22, 1, 1, 1, 1, 208],[8.2, 81, 0, 0, 1, 1,257]]
data_pandas = pd.DataFrame(data_prueba, columns = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour'])
data_prueba_norm = normalize(data_pandas)
dat = data_prueba_norm[['pH', 'Temprature', 'Taste', 'Odor', 'Fat ', 'Turbidity', 'Colour']].to_numpy()
print("Datos")
print(dat)

print("Resultados")
results = model.predict(dat).round()
print(results)
