# -*- coding: utf-8 -*-
"""
Created on Tue May  5 07:15:13 2020

@author: jcantero
"""

# Plantilla de Pre Procesado - Datos faltantes

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

# No es necesario el escalado de variables, ya que el modelo de regeresion realiza el escaldo de forma automatica

# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(xTrain, yTrain)

# Predecir el conjunto de test
yPred = regression.predict(xTest)

# Visualizar los resultados de entrenamiento
plt.scatter(xTrain, yTrain, color = "red")
plt.plot(xTrain, regression.predict(xTrain), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizar los resultados de test
plt.scatter(xTest, yTest, color = "red")
plt.plot(xTrain, regression.predict(xTrain), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()




