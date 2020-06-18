# LDA

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


# Aplicar LDA
library(MASS)
lda = lda(
  formula = Customer_Segment ~ .,
  data = trainingSet)

trainingSet = as.data.frame(predict(lda, trainingSet))
trainingSet = trainingSet[, c(5, 6, 1)]

testingSet = as.data.frame(predict(lda, testingSet))
testingSet = testingSet[, c(5, 6, 1)]


# Ajudtar el modelo de SVM con el conjunto de entrenamiento
library(e1071)
classifier = svm(formula = class ~ ., 
                 data = trainingSet,
                 type = "C-classification",
                 kernel = "linear")


# Predicion de los resultado con el conjunto de testing
# Probabilidad de prediccion
yPred = predict(classifier, newdata = testingSet[, -3])

# Matrix de confución
cm = table(testingSet[, 3], yPred)



# Visuliazacion del conjunto de entranmiento
library(ElemStatLearn)
set = trainingSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.025)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.025)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'LD1', ylab = 'LD2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue', 
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3]==2, 'blue3', 
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))


# VisualizaciÃÂ³n del conjunto de testing
set = testingSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.02)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Testing)',
     xlab = 'LD1', ylab = 'LD2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue', 
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3]==2, 'blue3', 
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))