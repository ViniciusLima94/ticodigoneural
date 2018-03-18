import numpy as np
import matplotlib.pyplot as plt
from InfoTheory import *

def NetworkGenerator(Nexc, pE, GE, Ninh, pI, GI):
	N = Nexc + Ninh
	A = np.zeros([N,N])

	for n in range(0, Nexc):
		for m in range(0, n) + range(n+1, N):
			if np.random.rand() <= pE:
				A[n, m] = GE

	for n in range(Nexc, N):
		for m in range(0, n) + range(n+1, N):
			if np.random.rand() <= pI:
				A[n, m] = GI
	return A

def BinomialNeuronNetwork(Sprevious, A, InProb):
	N = len(A)
	Prob_of_firing = np.zeros(N)
	Snext = np.zeros(len(Sprevious))


	for n in range(0, N):

		InputWeights = A[:, n]
		Prob_of_firing[n] = InProb[n] + np.sum(InputWeights*Sprevious)

		if Prob_of_firing[n] < 0:
			Prob_of_firing[n] = 0
		if Prob_of_firing[n] > 1.0:
			Prob_of_firing = 1.0

		dice = np.random.rand()

		if dice <= Prob_of_firing[n]:
			Snext[n] = 1
		else:
			Snext[n] = 0

	return Snext

# Parameters
Nexc = 100
pE = 0.01
GE = 0.3
Ninh = 0
pI = 0.0
GI = 0.0
# Integration time
T  = 5000
# Initial condition
Sinit = np.zeros(Nexc+Ninh)
# Constant input on all neurons 
InProb  = 0.01
InProbs = InProb*np.ones([Nexc+Ninh, T])

# Generating the network
Aexc = NetworkGenerator(Nexc, pE, GE, Ninh, pI, GI)

# Initializing the raster and setting initial conditions
Raster = np.zeros([Nexc+Ninh, T])
Raster[:,0] = Sinit

for t in range(1, T):
	Raster[:, t] = BinomialNeuronNetwork(Raster[:,t-1], Aexc, InProbs[:,t-1])

MI = np.zeros([100, 100])

for i in range(0, 100):
	for j in range(0,i)+range(i+1, 100):
		MI[i, j] = MutualInformation_Bin(Raster[i,:].astype(int), Raster[j,:].astype(int), 1)

