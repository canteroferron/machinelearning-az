# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:40:23 2020

@author: jcantero
"""
# Plantilla de Pre Procesado

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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoderX = LabelEncoder()
x[:, 0] = labelEncoderX.fit_transform(x[:, 0])

ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)

labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

# Dividir el dataset en conjunto de entranamiento y testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Escalar los datos, para tener el mimo ranfod e valores. La distancia de Euclides
from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
xTrain = xScaler.fit_transform(xTrain)
xTest = xScaler.transform(xTest)


