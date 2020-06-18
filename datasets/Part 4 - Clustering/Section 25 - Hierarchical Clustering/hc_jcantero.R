# Clusterting Jerarquico

# Importar los datos del centro comercial
dataset = read.csv("Mall_Customers.csv")
x = dataset[, 4:5]

# Utilizar el dendrograma para encontrar el nÃºmero Ã³ptimo de clusters
dendrogram = hclust(dist(x, method = "euclidean"),method = "ward.D")
plot(dendrogram,
     main = "Dendrograma",
     xlab = "Clientes del centro comercial",
     ylab = "Distancia Euclidea")

# Ajustar el clustering jerÃ¡rquico a nuestro dataset
hc = hclust(dist(x, method = "euclidean"), method = "ward.D")
yhc = cutree(hc, k=5)

# Visualizar los clusters
#install.packages("cluster")
library(cluster)
clusplot(x, 
         yhc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuacion (1-100)")