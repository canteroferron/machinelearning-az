# APriori

# Prepocesado de Datos


library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = TRUE)

summary(dataset)
itemFrequencyPlot(dataset, topN = 10)
itemFrequencyPlot(dataset, topN = 100)

# Entrenar algoritmo Apriori con el dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualizaccion de los resultados
inspect(sort(rules, by = 'lift')[1:10])