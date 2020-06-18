# Clustering con K-means

# Importar los datos
dataset = read.csv("Mall_Customers.csv")
x = dataset[, 4:5]

# MÃ©todo del codo
set.seed(6)
wcss = vector()
for (i in 1:10){
  wcss[i] <- sum(kmeans(x, i)$withinss)
}
plot(1:10, wcss, type = 'b', main = "Metodo del codo", xlab = "Numero de clusters (k)", ylab = "WCSS(k)")

# Aplicar el algoritmo de k-means con k Ã³ptimo
set.seed(29)
kmeans <- kmeans(x, 5, iter.max = 300, nstart = 10)

#VisualizaciÃ³n de los clusters
#install.packages("cluster")
library(cluster)
clusplot(x, 
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuación (1-100)")