# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:08:12 2020

@author: jcantero
"""

# Regresion linal multiple
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Codificacion de los datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

xLabelEncoder = LabelEncoder()
x[:, 3] = xLabelEncoder.fit_transform(x[:, 3])
oneHotEncoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
x = oneHotEncoder.fit_transform(x)

# Evitar la trnapa de la variable fisticia
x = x[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(xTrain, yTrain)

# Predicción de los resultados en el conjunto de testing
yPred = regression.predict(xTest)

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
# Se añade al inicio de la matrix un coluna de 1 en 50 filas, ncesario para añadir el temino
# independiante b0 y poder realizar la eliminacion hacia atras
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
sl = 0.05

xOpt = x[:, [0, 1, 2, 3, 4, 5]]
regresionOLS = sm.OLS(endog = y, exog = xOpt.tolist()).fit()
regresionOLS.summary()

# Eliminados la columna x2 ya que es la que mas p-valor tiene
xOpt = x[:, [0, 1, 3, 4, 5]]
regresionOLS = sm.OLS(endog = y, exog = xOpt.tolist()).fit()
regresionOLS.summary()

# Eliminados la columna x1 ya que es la que mas p-valor tiene
xOpt = x[:, [0, 3, 4, 5]]
regresionOLS = sm.OLS(endog = y, exog = xOpt.tolist()).fit()
regresionOLS.summary()

# Eliminados la columna x1 ya que es la que mas p-valor tiene
xOpt = x[:, [0, 3, 5]]
regresionOLS = sm.OLS(endog = y, exog = xOpt.tolist()).fit()
regresionOLS.summary()

# Eliminados la columna x1 ya que es la que mas p-valor tiene
xOpt = x[:, [0, 3]]
regresionOLS = sm.OLS(endog = y, exog = xOpt.tolist()).fit()
regresionOLS.summary()

# Nuestro modelo de datos solo contempla que la prediccion solo se considera el valor en S&D para predecir su beneficion
regresionOLS.predict(exog = [1,66051.52]);





