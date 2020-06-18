# Natural Language Processing

# Importar el data set
datasetOriginal = read.delim("Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)

# Limpieza de texto
library(tm)
library(SnowballC)

#Generar el corpus
corpus = VCorpus(VectorSource(datasetOriginal$Review))

# Transformar a minuscula
corpus = tm_map(corpus, content_transformer(tolower))

# Consultar el primer elemento
# as.character(corpus[[1]])

# Eliminación de numeros
corpus = tm_map(corpus, removeNumbers)

# Eliminacion de signo puntuación
corpus = tm_map(corpus, removePunctuation)

# Eliminación de palabras irrelevantes
corpus = tm_map(corpus, removeWords, stopwords(kind = "en"))

# Obtener las palabras raiz de cada palabra
corpus = tm_map(corpus, stemDocument)

# Eliminar espacios adicionales
corpus = tm_map(corpus, stripWhitespace)

# Creacion de la matriz dispersa de caracteristicas (Bag of Words)
dtm = DocumentTermMatrix(corpus)

# Eliminar las palabras menos frecuentes, ,e quedo con 99% de las palabras
dtm = removeSparseTerms(dtm, 0.999)

# Convertir a dataframe el dtm
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = datasetOriginal$Liked

# Coficicar la variable de clasificacion como factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Dividir el conjunto entre training y testing
library(caTools)
set.seed(123)
spit = sample.split(dataset$Liked, SplitRatio = 0.80)
trainingSet = subset(dataset, spit == TRUE)
testingSet = subset(dataset, spit == FALSE)


# Ajustar el modelo de regresion logistica con el conjunto de entrenamiento
#library(randomForest)
#classifier = randomForest(x = trainingSet[, -692], y = trainingSet$Liked, ntree = 10)
#yPred = predict(classifier, type = "response", newdata = testingSet[, -692])

# Ajudtar el modelo de regresion logistica con el conjunto de entrenamiento
#library(e1071)
#classifier = naiveBayes(x = trainingSet[, -692], y = trainingSet$Liked)
#yPred = predict(classifier, newdata = testingSet[, -692])


# Ajudtar el modelo de regresion logistica con el conjunto de entrenamiento
#library(e1071)
#classifier = svm(x = trainingSet[, -692], y = trainingSet$Liked, type = "C-classification", kernel = "linear")
#yPred = predict(classifier, type = "response", newdata = testingSet[, -692])

# Ajudtar el modelo de regresion logistica con el conjunto de entrenamiento y hacer las prediccion directamente con el conjunto de testing
library(class)
yPred = knn(train = trainingSet[, -692], test = testingSet[, -692], cl = trainingSet$Liked, k = 5) 



# Matrix de confuciÃ³n
library("caret")
cm = confusionMatrix(yPred, testingSet[, 692])
cm



