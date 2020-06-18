# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 23:08:24 2020

@author: jcantero
"""


# Convolution Neuras Networks

# Parte 1 - Construir el modelo de CNN

# Importar librerias y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Capa de convolución
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = "relu"))

# Paso 2 - Capa de Max Pooling
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Una segunda capa de convolucion y max pooling
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Una tercera capa de convolucion y max pooling al doble de filters. Esto suele emjorar la presición
classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Full connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid"))

# Compilar la CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Parte 2 - Ajustar la CNN a las imágenes para entrenar
# Realizar tranformaciones sobre las imagenes para entrenar com mas imagenes
# A las imagens se  le realizan trnadormacines de zoom, tamaños, se giran etc...
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_dataset = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

testing_dataset = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

classifier.fit(
        training_dataset,
        steps_per_epoch = 8000,
        epochs = 25,
        validation_data = testing_dataset,
        validation_steps = 2000)