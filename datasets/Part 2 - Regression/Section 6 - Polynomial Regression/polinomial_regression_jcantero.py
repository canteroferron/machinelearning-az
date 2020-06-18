# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:07:33 2020

@author: jcantero
"""

# Regresión polinomica

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Ajustar la regresion linear con el dataset
from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(x, y)

#Ajustar la refersion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
polynomialRegression = PolynomialFeatures(degree = 4)
xPoly = polynomialRegression.fit_transform(x)
linearRegression2 = LinearRegression()
linearRegression2.fit(xPoly, y)

# Visualización de los resultados del Modelo Lineal
plt.scatter(x, y, color = "red")
plt.plot(x, linearRegression.predict(x), color = "blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualización de los resultados del Modelo Polinómico
xGrid = np.arange(min(x), max(x), 0.1)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.scatter(x, y, color = "red")
plt.plot(xGrid, linearRegression2.predict(polynomialRegression.fit_transform(xGrid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Predicción de nuestros modelos
# Se ha añadido la sintaxis de doble corchete necesaria para hacer la predicción en las últimas versiones de Python (3.7+)
linearRegression.predict([[6.5]])
linearRegression2.predict(polynomialRegression.fit_transform([[6.5]]))