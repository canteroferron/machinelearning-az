#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:19:06 2019

@author: juangabriel
"""

# Plantilla de Pre Procesado - Datos Categóricos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoderX = LabelEncoder()
x[:, 0] = labelEncoderX.fit_transform(x[:, 0])

ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)

labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)