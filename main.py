import numpy as np
import matplotlib.pyplot as plt
import neuron_models as nm
import information_theory as it
from joblib import Parallel, delayed
import multiprocessing

'''
	RESAMPLE SPIKE TRAIN
'''
def resample(spike_train):
	lim1 = 0
	lim2 = 10
	idx  = 0
	spike_train_res = np.zeros(len(spike_train)/10.0)
	while lim2 <= len(spike_train):
		spike_train_res[idx] = np.mean(spike_train[lim1:lim2])
		lim1 = lim2
		lim2 += 10
		idx += 1
	return spike_train_res



def mirPOISSONneuron(seed_signal = 1, seed_noise = 2):
	poissonNeuron = nm.poissonNeuron(r0 = 100, epslon = 0.2)
	spike_train_poisson, signal_poisson = poissonNeuron.solve(seed_signal = seed_signal, seed_noise = seed_noise, tsim = 1e3, dt = 1.0e-3)
	_,_, mir_poisson = it.fMIR(spike_train_poisson, signal_poisson, poissonNeuron.S_t.dt)
	return mir_poisson


def mirLIFneuron(seed_signal = 1, seed_noise = 2):
	lifNeuron = nm.stochsticLIF(tau_m = 10.0e-3, mu = 0.75, D = 3.3e-3, c = 0.98, tabs = 0.0e-3)
	#lifNeuron = nm.stochsticLIF(tau_m = 10e-3, mu = 0.75, D = 6e-4, c = 0.34, tabs = 0.0e-3)
	spike_train_lif, signal_lif = lifNeuron.solve(seed_signal = seed_signal, seed_noise = seed_noise, tsim = 1e3, dt = .1e-3)
	_, _, mir_lif = it.fMIR(resample(spike_train_lif), resample(signal_lif), 10*lifNeuron.S_t.dt)
	return mir_lif

###############################################################################
# Frequency dependent mutual information rate for:
# Poisson Neuron Model
# Stochastic LIF Model
###############################################################################
Ntrials = 40

poissonNeuron = nm.poissonNeuron(r0 = 100, epslon = 0.2)
spike_train_poisson, signal_poisson = poissonNeuron.solve(seed_signal = i, seed_noise = Ntrials-i, tsim = 1e3, dt = 1.0e-3)
f_poisson, ilb_poisson, mir_poisson = it.fMIR(spike_train_poisson, signal_poisson, poissonNeuron.S_t.dt)


lifNeuron = nm.stochsticLIF(tau_m = 10.0e-3, mu = 0.75, D = 3.3e-3, c = 0.98, tabs = 0.0e-3)
lifNeuron = nm.stochsticLIF(tau_m = 10e-3, mu = 0.75, D = 6e-4, c = 0.34, tabs = 0.0e-3)
spike_train_lif, signal_lif = lifNeuron.solve(seed_signal = seed_signal, seed_noise = seed_noise, tsim = 1e3, dt = .1e-3)
spike_train_lif, signal_lif, mir_lif = it.fMIR(resample(spike_train_lif), resample(signal_lif), 10*lifNeuron.S_t.dt)
mir_l = Parallel(n_jobs=40)(delayed(mirLIFneuron)(seed_signal = i, seed_noise = Ntrials - 1) for i in range(Ntrials))
