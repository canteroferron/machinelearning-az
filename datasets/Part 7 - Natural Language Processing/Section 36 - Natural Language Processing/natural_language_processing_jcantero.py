# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:18:01 2020

@author: jcantero
"""

# Natural Leguaje Procesing

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Limpieza del texto
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Eliminad palabaras de la misma raiz: -ing, -ed
ps = PorterStemmer()

corpus = []

for i in range(0, len(dataset['Review'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    corpus.append(review)
    
# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)

x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Algoritmo de clasificacion para seleccioanr si la valoración es bueno o malo
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20, random_state = 0)

"""
# Naive Bayes
# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB();
"""

"""
# SVC
# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
"""

"""
# Desicion Tree
# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
"""

"""
# Random Forest
# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = "entropy", random_state = 0)
"""

# Maximum Entropy
# Ajustar el clasificador en el Conjunto de Entrenamiento
from nltk.classify import MaxentClassifier
classifier = MaxentClassifier(encoding = "utf8", weights = 2)



classifier.fit(xTrain, yTrain)

# Predicción de los resultados con el Conjunto de Testing
yPred  = classifier.predict(xTest)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)

# TP = #Verdaderos Positivos, TN =#Verdaderos Negativos, FP = #Falsos Positivos y FN = #Falsos Negativos
# Accuracy, la exactitud de la predicción: (TP+TN)/(TP+TN+FP+FN)
accuracy = (cm[1][1] + cm[0][0]) / cm.sum()

# Precision, medida de la precisión del algoritmo para la clase positiva: TP/(TP+FP)
presicionMedia = cm[1][1] / (cm[1][1] + cm[0][1])

# Recall, medida de la completitud del algoritmo: TP/(TP+FN)
recall = cm[1][1] / (cm[1][1] + cm[1][0])

# F1 Score, compromiso entre la precisión y la completitud: 2*Precision*Recall/(Precision+Recall)
f1Score = 2 * presicionMedia * recall / (presicionMedia + recall)


