# -*- coding: utf-8 -*-
"""
Created on Fri May  8 08:06:29 2020

@author: jcantero
"""

# Arboles de regresion
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Ajustar la regresión con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(x, y)

# Predicción de nuestros modelos
yPred = regression.predict([[6.5]])
print(yPred)

# Visualización de los resultados del Modelo Polinómico
xGrid = np.arange(min(x), max(x), 0.1)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.scatter(x, y, color = "red")
plt.plot(xGrid, regression.predict(xGrid), color = "blue")
plt.title("Modelo de Regresión Arboles de Desición")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
