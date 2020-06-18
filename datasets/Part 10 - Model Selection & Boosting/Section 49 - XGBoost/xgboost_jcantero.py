# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:45:41 2020

@author: jcantero
"""

# XGBOOST
# Cómo importar las librerías
import pandas as pd
import numpy as np

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoderX1 = LabelEncoder()
x[:, 1] = labelencoderX1.fit_transform(x[:, 1])
labelencoderX2 = LabelEncoder()
x[:, 2] = labelencoderX2.fit_transform(x[:, 2])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [2])],   
    remainder='passthrough'                        
)
x = np.array(ct.fit_transform(x), dtype=np.float)
x = x[:, 1:]


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo XGBOOST
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(xTrain, yTrain)

# Parte 3 Evaluar el modelo y calcular las predicciones finales
# Predicción de los resultados con el Conjunto de Testing
yPred  = classifier.predict(xTest)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = xTrain, y = yTrain, cv = 10)
accuracies.mean()
accuracies.std()
