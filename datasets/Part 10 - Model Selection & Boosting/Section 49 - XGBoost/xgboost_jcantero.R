# XGBOOST

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

# Ajustar XGBOSST al conjunto de entrenamiento
library(xgboost)
classifier = xgboost(data = as.matrix(trainingSet[, -11]), label = trainingSet$Exited, nrounds = 10)

# Aplicar F Fold Cross Validation
library(caret)
folds = createFolds(trainingSet$Exited, k = 10)
cv = lapply(folds, function(x) {
  trainingFold = trainingSet[-x, ]
  testingFold = trainingSet[x, ]
  
  classifier = xgboost(data = as.matrix(trainingSet[, -11]), label = trainingSet$Exited, nrounds = 10)
  
  yPred = predict(classifier, type = "response", newdata = as.matrix(testingFold[, -11]))
  yPred = (yPred >= 0.5)
  cm = table(testingFold[, 11], yPred)
  
  accurary = (cm[1, 1] + cm[2, 2]) / (cm[1, 1] + cm[2, 2] + cm[1, 2] + cm[2, 1])
  
  return(accurary)
})
accuracy = mean(as.numeric(cv))
accuracyStandardDeviation = sd(as.numeric(cv))

