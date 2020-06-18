# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:49:14 2020

@author: jcantero
"""

# K - Means

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargamos los datos con pandas
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values

# Método del codo para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kMeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kMeans.fit(x)
    wcss.append(kMeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()


# Aplicar el método de k-means para segmentar el data set
kMeans = KMeans(n_clusters = 5, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
yKMeans = kMeans.fit_predict(x)

# Visualización de los clusters
plt.scatter(x[yKMeans == 0, 0], x[yKMeans == 0, 1], s = 100, c = "red", label = "Cautos")
plt.scatter(x[yKMeans == 1, 0], x[yKMeans == 1, 1], s = 100, c = "blue", label = "Estandard")
plt.scatter(x[yKMeans == 2, 0], x[yKMeans == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(x[yKMeans == 3, 0], x[yKMeans == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(x[yKMeans == 4, 0], x[yKMeans == 4, 1], s = 100, c = "magenta", label = "Conservadores")
plt.scatter(kMeans.cluster_centers_[:,0], kMeans.cluster_centers_[:,1], s = 300, c = "yellow", label = "Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()