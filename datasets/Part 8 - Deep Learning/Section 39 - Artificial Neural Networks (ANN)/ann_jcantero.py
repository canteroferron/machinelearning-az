# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:12:11 2020

@author: jcantero
"""
# Redes neuronales atificiales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Parte 1 - Pre procesado de datos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Trandormamos la coluna 1 de la matriz de caracteriticas (Pais) en variables dummy
labelencoderX1 = LabelEncoder()
x[:, 1] = labelencoderX1.fit_transform(x[:, 1])

# Trandormamos la coluna 2 de la matrix de caracteristicas (Sexo) en variables dummy
labelencoderX2 = LabelEncoder()
x[:, 2] = labelencoderX2.fit_transform(x[:, 2])

# Facemos la trandormacion para generar las columasn de unos y ceros del pais
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
x = onehotencoder.fit_transform(x)
x = x[:, 1:]


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20, random_state = 0)


# Escalado de variables. En redes neuronales es obligatrio escalar las variables
from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
xTrain = xScaler.fit_transform(xTrain)
# Aqui aplicamos solo el transform, no el fit_tranform para que el escalado sea el mismo que se aplico al xTrain
xTest = xScaler.transform(xTest)


# Parte 2 - Contruir la red neuronal 
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential();

# Añadir la capas de entras y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))

# Añadir la segunda capa
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(xTrain, yTrain, batch_size = 10, epochs = 100)

# Parte 3 Evaluar el modelo y calcular las predicciones finales
# Predicción de los resultados con el Conjunto de Testing
yPred  = classifier.predict(xTest)

# Umbral de abandado del banco
yPred = (yPred > 0.5)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)



