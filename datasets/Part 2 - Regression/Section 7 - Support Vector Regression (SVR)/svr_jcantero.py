# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:39:49 2020

@author: jcantero
"""

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Escalado de variables
from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
yScaler = StandardScaler()
x = xScaler.fit_transform(x)
y = yScaler.fit_transform(y.reshape(-1,1))

# Ajustar la regresión con el dataset
from sklearn.svm import SVR
# Vamos a usar un kernel gausiano
regression = SVR(kernel = "rbf")
regression.fit(x, y)

# Predicción de nuestros modelos con SVR
yPred = yScaler.inverse_transform(regression.predict(xScaler.transform(np.array([[6.5]]))))

xWithoutScaler = xScaler.inverse_transform(x);
yWithoutScaler = yScaler.inverse_transform(y);

# Visualización de los resultados del SVR
xGrid = np.arange(min(xWithoutScaler), max(xWithoutScaler), 0.1)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.scatter(xWithoutScaler, yWithoutScaler, color = "red")
plt.plot(xGrid, yScaler.inverse_transform(regression.predict(xScaler.transform(xGrid))), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()