# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:14:30 2020

@author: jcantero
"""


# Plantilla de regresion 
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
xTrain = xScaler.fit_transform(xTrain)
xTest = xScaler.transform(xTest)"""

#Ajustar la regresion con el dataset
#Crer modelo de regresion

# Predicción de nuestros modelos
# Se ha añadido la sintaxis de doble corchete necesaria para hacer la predicción en las últimas versiones de Python (3.7+)
yPred = regression.predict([[6.5]]))


# Visualización de los resultados del Modelo Polinómico
xGrid = np.arange(min(x), max(x), 0.1)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.scatter(x, y, color = "red")
plt.plot(xGrid, regression.predict(xGrid), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

