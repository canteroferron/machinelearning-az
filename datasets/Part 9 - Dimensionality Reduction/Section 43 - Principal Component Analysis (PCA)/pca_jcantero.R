# ACP

# Importar el dataset
dataset = read.csv('Wine.csv')

# Dividir el conjunto entre training y testing
library(caTools)
set.seed(123)
spit = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
trainingSet = subset(dataset, spit == TRUE)
testingSet = subset(dataset, spit == FALSE)

# Escalar el dataset
trainingSet[, -14] = scale(trainingSet[, -14])
testingSet[, -14] = scale(testingSet[, -14])


# Proyeccion de las componentes principales
library(caret)
library(e1071)
pca = preProcess(x = trainingSet[, -14], method = "pca", pcaComp = 2)
trainingSet = predict(pca, trainingSet)

# Transposicion. reordenar la columana dependeicente pata que este la ultima
trainingSet = trainingSet[, c(2, 3, 1)]


testingSet = predict(pca, testingSet)

# Transposicion. reordenar la columana dependeicente pata que este la ultima
testingSet = testingSet[, c(2, 3, 1)]


# Ajudtar el modelo de regresion logistica con el conjunto de entrenamiento
classifier = svm(formula = Customer_Segment ~ ., 
                 data = trainingSet,
                 type = "C-classification",
                 kernel = "linear")


# Predicion de los resultado con el conjunto de testing
# Probabilidad de prediccion
yPred = predict(classifier, newdata = testingSet[, -3])

# Matrix de confuciÃ³n
cm = table(testingSet[, 3], yPred)



# Visuliazacion del conjunto de entranmiento
library(ElemStatLearn)
set = trainingSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.025)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.025)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue', 
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3]==2, 'blue3', 
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))


# VisualizaciÃ³n del conjunto de testing
set = testingSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.02)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Testing)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue', 
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3]==2, 'blue3', 
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))