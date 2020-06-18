# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:42:19 2020

@author: jcantero
"""

# Muestro Thompson (UCB)

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Algoritmo de UCB
import random
N = 10000
d = 10

numberOfRewards1 = [0] * d
numberOfRewards0 = [0] * d

# Anuncio seleccionado
adsSelected = [] 

totalReward = 0;

for n in range(0, N):
    maxRandom = 0
    ad = 0
    
    for i in range(0, d):
        randomBeta = random.betavariate(numberOfRewards1[i] + 1, numberOfRewards0[i] + 1)
            
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            ad = i
            
    adsSelected.append(ad)
    
    
    reward = dataset.values[n, ad]
    if reward == 1:
        numberOfRewards1[ad] = numberOfRewards1[ad] + 1;
    else:
        numberOfRewards0[ad] = numberOfRewards0[ad] + 1;
    
    totalReward = totalReward + reward
    
# Histograma de resultados
plt.hist(adsSelected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
