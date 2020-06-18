# Upper Confidence Bound

# Obtener el datasewt con la simucion de eventos, en este caso click en un anuncio en paticular ne una red social
dataset = read.csv("Ads_CTR_Optimisation.csv")

# Implementar UCB desde 0
N = 10000
d = 10
numberOfSelection = integer(d)
sumsOfRewards = integer(d)

adsSelected = integer(0)

totalRewards = 0;

for (n in 1:N) {
  maxUpperBound = 0
  ad = 0
  
  for (i in 1:d) {
    if (numberOfSelection[i] > 0) {
      averageReward = sumsOfRewards[i] / numberOfSelection[i]
      delta_i = sqrt(3 / 2 * log(n) / numberOfSelection[i])
    
      upperBound = averageReward + delta_i
    } else {
      upperBound = 1e400
    }
    
    if (upperBound > maxUpperBound) {
      maxUpperBound = upperBound
      ad = i
    }
  }
  
  adsSelected = append(adsSelected, ad)
  numberOfSelection[ad] = numberOfSelection[ad] + 1
  
  #Simulacion de click  en el anuncio
  reward = dataset[n, ad]
  sumsOfRewards[ad] = sumsOfRewards[ad] + reward
  
  totalRewards = totalRewards + reward
}

hist(
  adsSelected,
  col = "lightblue",
  main = "Histograma de los Anuncios",
  xlab = "ID del Anuncio",
  ylab = "Frecuencia absoluta del anuncio")

