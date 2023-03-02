#Elaborado por:
#César Argüello Salas
#Jose Fabio Navarro Naranjo
#------------------------------------------------------------------------------
#Importación de librerías necesarias
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#No mostrar warnings en el output
warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------
#Descargar el conjunto de imágenes desde una carpeta de Dropbox personal
dataset_url = "https://www.dropbox.com/s/6v5v9275fntr18k/casting_data.zip?dl=1" #Enlace de descarga

#Función que realiza la descarga, extracción y guardado de las imágenes en la ruta /content/datasets
get_file(origin=dataset_url,
         fname='casting_data',
         cache_dir='/content',
         extract=True)

#------------------------------------------------------------------------------
#Generar los datasets de entrenamiento, validación y prueba

#Hiperparámetros de esta sección
batch_size = 32 #Tamaño de lote (batch)
validation_split = 0.2 #Porcentaje de datos usados para validación

########### USANDO LOS SUBCONJUNTOS PREDETERMINADOS POR LOS AUTORES ###########
# (Comentar esta sección si no se utilizan)
#Los subconjuntos de entrenamiento/validación se extraen de la carpeta 'train'
#El subconjunto de prueba se extrae de la carpeta 'test'

seed = np.random.randint(1000) #Genera un entero aleatorio para emplear
        #como semilla para el split de datos de entrenamiento/validación

#Cargado de las imágenes en datasets mediante
#la función 'image_dataset_from_directory'

train_ds = image_dataset_from_directory( #subconjunto de entrenamiento
    '/content/datasets/train',
    labels='inferred',
    label_mode='binary',
    class_names=['ok_front', 'def_front'],
    color_mode='grayscale',
    image_size=(300, 300),
    batch_size=batch_size,
    shuffle=True,
    seed=seed,
    validation_split=validation_split,
    subset='training')

val_ds = image_dataset_from_directory( #subconjunto de validación
    '/content/datasets/train',
    labels='inferred',
    label_mode='binary',
    class_names=['ok_front', 'def_front'],
    color_mode='grayscale',
    image_size=(300, 300),16
    batch_size=batch_size,
    shuffle=True,
    seed=seed,
    validation_split=validation_split,
    subset='validation')

test_ds = image_dataset_from_directory( #subconjunto de prueba
    '/content/datasets/test',
    labels='inferred',
    label_mode='binary',
    class_names=['ok_front', 'def_front'],
    color_mode='grayscale',
    image_size=(300, 300),
    batch_size=batch_size,
    shuffle=True)

##################### USANDO SUBCONJUNTOS PERSONALIZADOS ######################
# (Comentar esta sección si no se utilizan)
"""
#El conjunto de imágenes completo se extrae de la carpeta 'custom'

#Se definen dos listas con los labels de cada grupo de imágenes
ok_labels = list(np.zeros(3137, dtype=int)) #lista de ceros para 'OK'
def_labels = list(np.ones(4211, dtype=int)) #lista de unos para 'Defective'

#Cargado del conjunto en dos datasets para cada tipo
ok_ds = image_dataset_from_directory( #dataset de imágenes 'OK'
    '/content/datasets/custom/ok_front',
    labels=ok_labels, #se etiquetan con 0
    label_mode='int',
    color_mode='grayscale',
    image_size=(300, 300),
    batch_size=None,
    shuffle=True)

def_ds = image_dataset_from_directory( #dataset de imágenes 'Defective'
    '/content/datasets/custom/def_front',
    labels=def_labels, #se etiquetan con 1
    label_mode='int',
    color_mode='grayscale',
    image_size=(300, 300),
    batch_size=None,17
    shuffle=True)

#Se genera el subconjunto de entrenamiento con:
train_ds = ok_ds.skip(262) #2875 imágenes 'OK'
train_ds = train_ds.concatenate(def_ds.skip(453)) #3758 imágenes 'Defective'
train_ds = train_ds.shuffle(6633, reshuffle_each_iteration=False) #barajado

#Se genera el subconjunto de prueba con:
test_ds = ok_ds.take(262) #262 imágenes 'OK'
test_ds = test_ds.concatenate(def_ds.take(453)) #453 imágenes 'Defective'
test_ds = test_ds.shuffle(715, reshuffle_each_iteration=False) #barajado

#Se extrae un subconjunto de 'train_ds' para validación
val_size = int(6633 * validation_split)
val_ds = train_ds.take(val_size)
train_ds = train_ds.skip(val_size)

#Se dividen los datasets en lotes (batches)
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)
"""
###############################################################################

#Configuración de los subconjuntos de datos para mejor rendimiento
AUTOTUNE = tf.data.AUTOTUNE

#Mantiene los datos en cache luego de cargarlos y superpone las tareas de
#preprocesamiento de datos y ejecución del modelo durante el entrenamiento
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#------------------------------------------------------------------------------
#Definición del modelo de CNN
#Hiperparámetros de las capas convolucionales:
#filters            #número de filtros (kernels) por capa
#kernel_size        #tamaño del kernel (dimensión NxN)
#strides            #tamaño de paso entre deslizamientos18

#Hiperparámetros de las capas de pooling:
#pool_size          #tamaño de la ventana de pooling (dimensión NxN)
#strides            #tamaño de paso entre deslizamientos

#Hiperparámetros de las capas densas:
#units              #número de neuronas por capa
dropout_rate = 0    #porcentaje de dropout

model = keras.Sequential([
    
    #Capa de preprocesamiento: normaliza los datos al rango [0, 1]
    layers.Rescaling(scale=1./255, input_shape=(300, 300, 1)),

    #Primera capa de convolución
    layers.Conv2D(filters=8,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  activation='relu'),
    
    #Primera capa de pooling
    layers.MaxPool2D(pool_size=2,
                     strides=2,
                     padding='same'),
    
    #Segunda capa de convolución
    layers.Conv2D(filters=8,
                  kernel_size=5,
                  strides=3,
                  padding='same',
                  activation='relu'),
    
    #Segunda capa de pooling
    layers.MaxPool2D(pool_size=5,
                     strides=5,
                     padding='same'),
    
    #Operación de flattening (convertir a vector)
    layers.Flatten(),
    
    #Primera capa densa19
    layers.Dense(units=25, activation='sigmoid'),

    #Implementación de dropout a primera capa densa
    layers.Dropout(rate=dropout_rate),

    #Capa de salida
    layers.Dense(units=1, activation='sigmoid')
])

model.summary() #imprime un resumen del modelo creado

#------------------------------------------------------------------------------
#Compilación y entrenamiento del modelo

model.compile( #compilación: configura el entrenamiento del modelo
    optimizer='adam', #se usa el optimizador Adam
    loss='binary_crossentropy', #función de pérdida empleada
    metrics=['accuracy']) #métrica a ser evaludada: exactitud

#función para detener el entrenamiento de manera prematura
early_stopping = EarlyStopping( #se usa para evitar el sobreentrenamiento
    monitor='val_loss', #variable a monitorear: pérdida de validación
    patience=2, #número máximo de epochs sin que haya mejora
    restore_best_weights=True, #permite restaurar los pesos al mejor epoch
    verbose=1) #muestra el resultado en el output

history = model.fit( #ejecuta el entrenamiento del modelo
    train_ds, #dataset de entrenamiento
    validation_data=val_ds, #dataset de validación
    epochs=40, #número máximo de epochs
    callbacks=[early_stopping], #llamada a la función de EarlyStopping
    verbose=1) #muestra el resultado en el output

#------------------------------------------------------------------------------
#Display de resultados del proceso de entrenamiento y validación

#conversión del historial de resultados a un dataframe
history_df = pd.DataFrame(history.history)

#se le suma 1 a cada índice para que coincidan con el número de epoch
history_df.index = map(lambda x : x+1, history_df.index)20

#display de las curvas de pérdida de entrenamiento/validación
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(['train loss', 'val loss'])
plt.show()

#display de las curvas de exactitud de entrenamiento/validación
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(['train accuracy', 'val accuracy'])
plt.show()

#obtención de variables en la epoch con menor pérdida de validación
best_epoch = history_df['val_loss'].idxmin()
t_loss = history_df.loc[best_epoch, 'loss']
v_loss = history_df.loc[best_epoch, 'val_loss']
t_acc = history_df.loc[best_epoch, 'accuracy']
v_acc = history_df.loc[best_epoch, 'val_accuracy']

#impresión de resultados de la epoch con menor pérdida de validación
print("--------------------------------")
print("Best validation loss at epoch", best_epoch, "\n")
print("Training loss: {:.2%}".format(t_loss))
print("Validation loss: {:.2%}".format(v_loss))
print("Training accuracy: {:.2%}".format(t_acc))
print("Validation accuracy: {:.2%}".format(v_acc))
print("--------------------------------")

#------------------------------------------------------------------------------
#Prueba del modelo mediante predicción con nuevos datos
#se ejecuta el modelo con el dataset de prueba
#y se guardan los resultados en un array
pred_probability = model.predict(test_ds)

#se evalúan los resultados de la clasificación
predictions = pred_probability > 0.5

#si la probabilidad de que esté defectuoso
#supera un 50% se clasifica como defectuoso (1)21
#de lo contrario se clasifica como OK (0)
#se obtienen las etiquetas reales de las imágenes de prueba
#y se almacenan en un array
true_labels = np.concatenate([y for (x, y) in test_ds], axis=0)

#display de la matriz de confusión
plt.figure(figsize=(4,3))
plt.title('Confusion Matrix', size=20, weight='bold')
sns.heatmap(
    confusion_matrix(true_labels, predictions),
    annot=True,
    annot_kws={'size':14, 'weight':'bold'},
    fmt='d',
    cbar=False,
    cmap='YlGnBu',
    xticklabels=['OK', 'Defective'],
    yticklabels=['OK', 'Defective'])
plt.tick_params(axis='both', labelsize=14)
plt.ylabel('Actual', size=14, weight='bold')
plt.xlabel('Predicted', size=14, weight='bold')
plt.show()

#impresión del reporte de clasificación
print("\n-----------------------------------------------------")
print("Classification report:\n")
print(classification_report(true_labels, predictions, digits=4,
    target_names=['OK', 'Defective']))
print("-----------------------------------------------------")
