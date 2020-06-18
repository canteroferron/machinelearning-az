#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:19:21 2019

@author: juangabriel
"""

# Plantilla de Pre Procesado - Datos faltantes

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


from sklearn.impute import SimpleImputer
# Reemplazar por medias
imputer = SimpleImputer(strategy="mean")
#medias en columnas 1,2
imputer = imputer.fit(x[:, 1:3])
# Cambiar valores por dichas nedias
x[:, 1:3] = imputer.transform(x[:, 1:3])