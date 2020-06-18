# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:42:33 2020

@author: jcantero
"""


# Clustering Jerárquico

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar los datos del centro comercial con pandas
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values


# Utilizar el dendrograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# Ajustar el clustetring jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
yhc = hc.fit_predict(x)


# Visualización de los clusters
plt.scatter(x[yhc == 0, 0], x[yhc == 0, 1], s = 100, c = "red", label = "Cautos")
plt.scatter(x[yhc == 1, 0], x[yhc == 1, 1], s = 100, c = "blue", label = "Estandard")
plt.scatter(x[yhc == 2, 0], x[yhc == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(x[yhc == 3, 0], x[yhc == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(x[yhc == 4, 0], x[yhc == 4, 1], s = 100, c = "magenta", label = "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()