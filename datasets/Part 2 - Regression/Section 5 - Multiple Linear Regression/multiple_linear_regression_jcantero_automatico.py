# -*- coding: utf-8 -*-
"""
Created on Wed May  6 07:23:45 2020

@author: jcantero
"""

# Regresion linal multiple
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Codificacion de los datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

xLabelEncoder = LabelEncoder()
x[:, 3] = xLabelEncoder.fit_transform(x[:, 3])
oneHotEncoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
x = oneHotEncoder.fit_transform(x)

# Evitar la trnapa de la variable fisticia
x = x[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(xTrain, yTrain)

# Predicción de los resultados en el conjunto de testing
yPred = regression.predict(xTest)


# Se añade al inicio de la matrix un coluna de 1 en 50 filas, ncesario para añadir el temino
# independiante b0 y poder realizar la eliminacion hacia atras
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
def backwardElimination(xToEliminate, slToCompare):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    
    for i in range(0, numVars):
        regressorOLS = sm.OLS(endog = y, exog = xToEliminate.tolist()).fit()
        maxVar = max(regressorOLS.pvalues).astype(float)        
        adjRBefore = regressorOLS.rsquared_adj.astype(float)
        
        if maxVar > slToCompare:
            for j in range(0, numVars - i):
                if (regressorOLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = xToEliminate[:, j]
                    xToEliminate = np.delete(xToEliminate, j, 1)
                    
                    tmpRegressorOLS = sm.OLS(endog = y, exog = xToEliminate.tolist()).fit()
                    adjRAfter = tmpRegressorOLS.rsquared_adj.astype(float)
                    
                    if (adjRBefore >= adjRAfter):                        
                        xRollback = np.hstack((xToEliminate, temp[:, [0,j]]))                        
                        xRollback = np.delete(xRollback, j, 1)     
                        print (regressorOLS.summary())                        
                        return xRollback                    
                    else:                        
                        continue
    regressorOLS.summary()    
    return x 
                    
sl = 0.05

xOpt = x[:, [0, 1, 2, 3, 4, 5]]
xModeled = backwardElimination(xOpt, sl)

# Ya tenemos en xModeled los x que han qdado despues de la eliminacion hacia atras, con ellas creamos el modelo
regresion = sm.OLS(endog = y, exog = xModeled.tolist()).fit()
regresion.predict(exog = [1, 66051.52, 25000])