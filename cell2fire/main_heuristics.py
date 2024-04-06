import random
import numpy as np
from ga import simulate_ga
from grasp import simulate_grasp
seeds=list(range(1,5)) # seeds
archivo="Exp100x100_hetero_C1" #file of landscape de bosque a simular
maxTime=7200
ntest=len(seeds) #numero de pruebas
percentages=[0.01,0.03,0.05,0.075,0.1,0.125,0.15]
for perc in percentages:
	for i in range(ntest):
		random.seed(seeds[i])
		np.random.seed(seeds[i])
		simulate_grasp(archivo,i,maxTime,perc)
		simulate_ga(archivo,i,perc,maxTime)


