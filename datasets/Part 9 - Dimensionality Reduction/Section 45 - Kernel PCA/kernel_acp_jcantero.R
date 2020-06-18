# Kernel ACP

# Importar el dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]



# Dividir el conjunto entre training y testing
library(caTools)
set.seed(123)
spit = sample.split(dataset$Purchased, SplitRatio = 0.75)
trainingSet = subset(dataset, spit == TRUE)
testingSet = subset(dataset, spit == FALSE)

# Escalar el dataset
trainingSet[, 1:2] = scale(trainingSet[, 1:2])
testingSet[, 1:2] = scale(testingSet[, 1:2])


# Aplicar kernel
library(kernlab)
kpca = kpca(~., data = trainingSet[, -3], kernel = "rbfdot", features = 2)
trainingSetPCA = as.data.frame(predict(kpca, trainingSet))
testingSetPCA = as.data.frame(predict(kpca, testingSet))

trainingSetPCA$Purchased = trainingSet$Purchased
testingSetPCA$Purchased = trainingSet$Purchased



# Ajudtar el modelo de regresion logistica con el conjunto de entrenamiento
classifier = glm(formula = Purchased ~ ., data = trainingSetPCA, family = binomial)


# Predicion de los resultado con el conjunto de testing
# Probabilidad de prediccion
probPred = predict(classifier, type = "response", newdata = testingSetPCA[, -3])

# Establecer la probabilidad en 0 o 1
yPred = ifelse(probPred > 0.5, 1, 0)

# Matrix de confuciÃ³n
cm = table(testingSet[, 3], yPred)



# Visuliazacion del conjunto de entranmiento
library(ElemStatLearn)
set = trainingSetPCA
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
gridSet = expand.grid(x1, x2)
colnames(gridSet) = c('V1', 'V2')
probSet = predict(classifier, type = 'response', newdata = gridSet)
yGrid = ifelse(probSet > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Entrenamiento)',
     xlab = 'V1', ylab = 'V2',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(yGrid), length(x1), length(x2)), add = TRUE)
points(gridSet, pch = '.', col = ifelse(yGrid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


set = testingSetPCA
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
gridSet = expand.grid(x1, x2)
colnames(gridSet) = c('V1', 'V2')
probSet = predict(classifier, type = 'response', newdata = gridSet)
yGrid = ifelse(probSet > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Testing)',
     xlab = 'V1', ylab = 'V2',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(yGrid), length(x1), length(x2)), add = TRUE)
points(gridSet, pch = '.', col = ifelse(yGrid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))