# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:25:47 2020

@author: jcantero
"""


# Bosques de regresion aleatorios
# Plantilla de regresion 
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Ajustar el Random Forest con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 300, random_state = 0)
regression.fit(x, y)

# Predicción de nuestros modelos con Random Forest
yPred = regression.predict([[6.5]])

# Visualización de los resultados del Modelo Polinómico
xGrid = np.arange(min(x), max(x), 0.01)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.scatter(x, y, color = "red")
plt.plot(xGrid, regression.predict(xGrid), color = "blue")
plt.title("Modelo de Regresión Random Forest")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

