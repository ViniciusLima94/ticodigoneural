import numpy as np
import matplotlib.pyplot as plt

def BinNeuronEntropy(SpikeTrain):
	T = len(SpikeTrain)
	P_firing = np.sum(SpikeTrain) / float(T)
	P_notfiring = 1.0 - P_firing

	return -P_firing*np.log2(P_firing) - P_notfiring*np.log2(P_notfiring)

def EntropyFromProbabilities(Prob):
	H = 0
	s = Prob.shape
	for p in Prob:
		if p > 0.00001:
			H -= p*np.log2(p)
	return H

def MutualInformation_Bin(sX, sY, tau):
	PX = np.zeros([2])
	PY = np.zeros([2])
	PXY = np.zeros([2,2])
	for t in range( np.maximum(0, 0-tau), np.minimum(len(sX)-tau, len(sX)) ):
		PX[sX[t]] += 1
		PY[sY[t+tau]] += 1
		PXY[sX[t], sY[t+tau]] += 1

	# Normalizing probabilities
	PX = PX / np.sum(PX)
	PY = PY / np.sum(PY)
	PXY = PXY / np.sum(PXY)
	HX = EntropyFromProbabilities(PX)
	HY = EntropyFromProbabilities(PY)
	HXY = EntropyFromProbabilities(np.reshape(PXY, (4)))
	MI  = HX + HY - HXY
	return MI
