#Importar el data set
dataset = read.csv('Data.csv')

#Tratamiento de los dataset
dataset$Age = 
  ifelse(
    is.na(dataset$Age), 
    ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
    dataset$Age)

dataset$Salary = 
  ifelse(
    is.na(dataset$Salary), 
    ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
    dataset$Salary)

# Codificar las variables categoricas
dataset$Country = 
  factor(
    dataset$Country,
    levels = c("France", "Spain", "Germany"),
    labels = c(1, 3, 3))

dataset$Purchased = 
  factor(
    dataset$Purchased,
    levels = c("No", "Yes"),
    labels = c(0, 1))

# Dividir los datos en conjunto de entrenamiento y test
#install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testingSet = subset(dataset, split == FALSE)

# Escalar los datos, para tener el mimo ranfod e valores. La distancia de Euclides
trainingSet[, 2:3] = scale(trainingSet[, 2:3])
testingSet[, 2:3] = scale(testingSet[, 2:3])
