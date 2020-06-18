# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:32:53 2020

@author: jcantero
"""

# Upper Confidence Bound (UCB)

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Algoritmo de UCB
import math
N = 10000
d = 10
numberOfSelections = [0] * d
sumOfRewards = [0] * d

# Anuncio seleccionado
adsSelected = [] 

totalReward = 0;

for n in range(0, N):
    maxUpperBound = 0
    ad = 0
    
    for i in range(0, d):
        if (numberOfSelections[i] > 0):
        
            averageReward = sumOfRewards[i] / numberOfSelections[i]
            delta_i = math.sqrt(3 /2 * math.log(n + 1) /  numberOfSelections[i])
        
            upperBound = averageReward + delta_i
        else:
            upperBound = 1e400
            
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            ad = i
            
    adsSelected.append(ad)
    numberOfSelections[ad] = numberOfSelections[ad] + 1 
    
    reward = dataset.values[n, ad]
    sumOfRewards[ad] = sumOfRewards[ad] + reward
    
    totalReward = totalReward + reward
    
# Histograma de resultados
plt.hist(adsSelected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
    

