# Muestro de thompson

# Obtener el datasewt con la simucion de eventos, en este caso click en un anuncio en paticular ne una red social
dataset = read.csv("Ads_CTR_Optimisation.csv")

# Implementar UCB desde 0
N = 10000
d = 10

numberOfRewards1 = integer(d)
numberOfRewards0 = integer(d)

adsSelected = integer(0)

totalRewards = 0;

for (n in 1:N) {
  maxRandom = 0
  ad = 0
  
  for (i in 1:d) {
    randomBeta = rbeta(n = 1, shape1 = numberOfRewards1[i] + 1, shape2 = numberOfRewards0[i] + 1)
    
    
    if (randomBeta > maxRandom) {
      maxRandom = randomBeta
      ad = i
    }
  }
  
  adsSelected = append(adsSelected, ad)
  
  
  #Simulacion de click  en el anuncio
  reward = dataset[n, ad]
  if (reward == 1) {
    numberOfRewards1[ad] = numberOfRewards1[ad] + 1  
  } else {
    numberOfRewards0[ad] = numberOfRewards0[ad] + 1
  }
 
  
  totalRewards = totalRewards + reward
}

hist(
  adsSelected,
  col = "lightblue",
  main = "Histograma de los Anuncios",
  xlab = "ID del Anuncio",
  ylab = "Frecuencia absoluta del anuncio")

