import numpy as np
import scipy.signal as spy
import scipy.integrate as sint

r'''
    Calculates frequency dependent mutual information rate (fMIR)
    > spike_train: Neuron spike train
    > signal:      Stimulus signal
    > dt:          Time resolution
'''
def fMIR(spike_train = None, signal = None, dt = None):
    f, cxy = spy.coherence(spike_train, signal, fs = 1.0/dt)
    ilb = -np.log2(1-cxy)
    MIR = 1000*sint.simps(y = ilb, x = f)
    return 1000*f, ilb, MIR
