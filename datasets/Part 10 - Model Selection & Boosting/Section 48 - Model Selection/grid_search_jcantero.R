# Grid search

# Importar el dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

dataset$Purchased = factor(dataset$Purchased)

# Dividir el conjunto entre training y testing
library(caTools)
set.seed(123)
spit = sample.split(dataset$Purchased, SplitRatio = 0.75)
trainingSet = subset(dataset, spit == TRUE)
testingSet = subset(dataset, spit == FALSE)

# Escalar el dataset
trainingSet[, 1:2] = scale(trainingSet[, 1:2])
testingSet[, 1:2] = scale(testingSet[, 1:2])

# Ajudtar el modelo de regresion logistica con el conjunto de entrenamiento
library(e1071)
classifier = svm(formula = Purchased ~ ., data = trainingSet, type = "C-classification", kernel = "radial")


# Predicion de los resultado con el conjunto de testing
# Probabilidad de prediccion
yPred = predict(classifier, type = "response", newdata = testingSet[, -3])


# Matrix de confucion
cm = table(testingSet[, 3], yPred)

# Aplicar F Fold Cross Validation
library(caret)
folds = createFolds(trainingSet$Purchased, k = 10)
cv = lapply(folds, function(x) {
  trainingFold = trainingSet[-x, ]
  testingFold = trainingSet[x, ]
  
  classifier = svm(formula = Purchased ~ ., data = trainingFold, type = "C-classification", kernel = "radial")
  
  yPred = predict(classifier, type = "response", newdata = testingFold[, -3])
  cm = table(testingFold[, 3], yPred)
  
  accurary = (cm[1, 1] + cm[2, 2]) / (cm[1, 1] + cm[2, 2] + cm[1, 2] + cm[2, 1])
  
  return(accurary)
})
accuracy = mean(as.numeric(cv))
accuracyStandardDeviation = sd(as.numeric(cv))


# Aplicar grid search para encontrar los parametros optimos
library(caret)
classifier = train(form = Purchased ~ ., data = trainingSet, method = 'svmRadial')



# Visuliazacion del conjunto de entranmiento
library(ElemStatLearn)
set = trainingSet
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
gridSet = expand.grid(x1, x2)
colnames(gridSet) = c('Age', 'EstimatedSalary')
yGrid = predict(classifier, newdata = gridSet)
plot(set[, -3],
     main = 'Kernel CVM (Conjunto de Entrenamiento)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(yGrid), length(x1), length(x2)), add = TRUE)
points(gridSet, pch = '.', col = ifelse(yGrid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


set = testingSet
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
gridSet = expand.grid(x1, x2)
colnames(gridSet) = c('Age', 'EstimatedSalary')
yGrid = predict(classifier, newdata = gridSet)
plot(set[, -3],
     main = 'Kernel SVM (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(yGrid), length(x1), length(x2)), add = TRUE)
points(gridSet, pch = '.', col = ifelse(yGrid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
