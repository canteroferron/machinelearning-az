# Redes Neuronales Artificiales



# Importar el dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[, 4:14]

# Codificar los factores para la red neuronal
dataset$Geography = 
  as.numeric(
    factor(
      dataset$Geography,
      levels = c("France", "Spain", "Germany"),
      labels = c(1, 2, 3)))

dataset$Gender = 
  as.numeric(
    factor(dataset$Gender,
    levels = c("Female", "Male"),
    labels = c(1,2)))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.80)
trainingSet = subset(dataset, split == TRUE)
testingSet = subset(dataset, split == FALSE)

# Escalado de valores. Es obligatorio en redes neuronales
trainingSet[, -11] = scale(trainingSet[, -11])
testingSet[, -11] = scale(testingSet[, -11])

# Ajustar el clasificador con el conjunto de entrenamiento.
# Crear la red neuronal
library(h2o)

# Podemos conectaros a instancias de h2o, por ejemplo en amazon para usar comnpurtación en la nube
# Todos lso cores menos 1
Sys.setenv(JAVA_HOME="D:/Programas/ProgramasNoInstalados/java/jre1.8.0_241")
h2o.init(nthreads = -1);

classifier = 
  h2o.deeplearning(
    y = "Exited",
    training_frame = as.h2o(trainingSet),
    activation = "Rectifier",
    hidden = c(6, 6),
    epochs = 100,
    train_samples_per_iteration = -2)

# Predicción de la probalilidad de que un cliente abandone el banco
proPred = h2o.predict(classifier, newdata = as.h2o(testingSet[, -11]))

# Mas del 50% de dejar el banco se marca como que lo va a dejar
yPred = (proPred > 0.5)

# Transformacion a vectores
yPred = as.vector(yPred)

# Crear la matriz de confusión
cm = table(testingSet[, 11], yPred)

# Cerrar la sesion de h2o
h2o.shutdown()
