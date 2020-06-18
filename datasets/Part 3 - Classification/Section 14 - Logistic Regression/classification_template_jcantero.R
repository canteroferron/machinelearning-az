# Pantilla de clasifisificacion

# Importar el dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Coficicar la variable de clasificacion como factor
#dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

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



# Predicion de los resultado con el conjunto de testing
# Probabilidad de prediccion
yPred = predict(classifier, type = "response", newdata = testingSet[, -3])


# Matrix de confución
cm = table(testingSet[, 3], yPred)


# Visuliazacion del conjunto de entranmiento
library(ElemStatLearn)
set = trainingSet
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
gridSet = expand.grid(x1, x2)
colnames(gridSet) = c('Age', 'EstimatedSalary')
yGrid = predict(classifier, newdata = gridSet)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Entrenamiento)',
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
     main = 'Clasificación (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(yGrid), length(x1), length(x2)), add = TRUE)
points(gridSet, pch = '.', col = ifelse(yGrid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


