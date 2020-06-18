# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 07:58:58 2020

@author: jcantero
"""

#Kernel ACP

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
xTrain = xScaler.fit_transform(xTrain)
xTest = xScaler.transform(xTest)

# Aplicar ACP
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = "rbf")
xTrain = kpca.fit_transform(xTrain)
xTest = kpca.transform(xTest)

# Ajustar el modelo de Regresión Logística en el Conjunto de Entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xTrain, yTrain)

# Predicción de los resultados con el Conjunto de Testing
yPred  = classifier.predict(xTest)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)

# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
xSet, ySet = xTrain, yTrain
x1, x2 = np.meshgrid(
            np.arange(start = xSet[:, 0].min() - 1, stop = xSet[:, 0].max() + 1, step = 0.01),
            np.arange(start = xSet[:, 1].min() - 1, stop = xSet[:, 1].max() + 1, step = 0.01))
plt.contourf(
    x1, 
    x2, 
    classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
    alpha = 0.75, 
    cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(
        xSet[ySet == j, 0], 
        xSet[ySet == j, 1],
        c = ListedColormap(('red', 'green'))(i), 
        label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
xSet, ySet = xTest, yTest
x1, x2 = np.meshgrid(
    np.arange(start = xSet[:, 0].min() - 1, stop = xSet[:, 0].max() + 1, step = 0.01),
    np.arange(start = xSet[:, 1].min() - 1, stop = xSet[:, 1].max() + 1, step = 0.01))
plt.contourf(
    x1, 
    x2, 
    classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
    alpha = 0.75, 
    cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(
        xSet[ySet == j, 0], 
        xSet[ySet == j, 1],
        c = ListedColormap(('red', 'green'))(i), 
        label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()