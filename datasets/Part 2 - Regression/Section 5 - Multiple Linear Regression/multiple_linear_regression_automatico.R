# Regresion lineal multiple automatica

#Importar el data set
dataset = read.csv('50_Startups.csv')

dataset$State = 
  factor(
    dataset$State,
    levels = c("New York", "California", "Florida"),
    labels = c(1, 2, 3))

# Dividir los datos en conjunto de entrenamiento y test
#install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testingSet = subset(dataset, split == FALSE)

# Alinear el modelo de Ragresion Linal Multiple con el conjunto de Entrenamiento
# Crea el modelos lineal como variable dependiente y el punto son todas las demas independientes
regression = lm(formula = Profit ~ ., data = trainingSet)

# Predecir los resultados con el conjunto de testing
yPred = predict(regression, newdata = testingSet)

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)) {
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

sl = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(dataset, sl)
