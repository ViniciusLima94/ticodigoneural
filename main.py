import numpy as np
import matplotlib.pyplot as plt
import neuron_models as nm
import information_theory as it

###############################################################################
# Frequency dependent mutual information rate for:
# Poisson Neuron Model
# Stochastic LIF Model
###############################################################################
poissonNeuron = nm.poissonNeuron(r0 = 100, epslon = 0.2)
spike_train_poisson, signal_poisson = poissonNeuron.solve(seed_signal = 1, seed_noise = 2, tsim = 1e6, dt = 1.0)
f_poisson, ilb_poisson, mir_poisson = it.fMIR(spike_train_poisson, signal_poisson, poissonNeuron.S_t.dt)
#
lifNeuron = nm.stochsticLIF(tau_m = 10.0, mu = 0.75, D = 3.3, c = 0.98, tabs = 0.0)
spike_train_lif, signal_lif = lifNeuron.solve(seed_signal = 1, seed_noise = 2, tsim = 1e6, dt = 1.0)
f_lif, ilb_lif, mir_lif = it.fMIR(spike_train_lif, signal_lif, lifNeuron.S_t.dt)
