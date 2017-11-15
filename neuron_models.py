import numpy as np
import signal_model as sm

r'''
    Poisson Neuron Model
'''
class poissonNeuron(object):

    r'''
        Constructor
        > r0:     Mean frequency of the poisson generator (default 100 Hz)
        > epslon: Signal strength                         (defalt  0.2)
    '''
    def __init__(self, r0 = 100, epslon = 0.2):
        self.r0 = r0
        self.epslon = epslon

    def model_equation(self, ):
        None

    r'''
        Solve model equations
        tsim: Simulation time
        dt:   Time resolution
    '''
    def solve(self, seed_signal = 1, seed_noise = 2, tsim = 1000, dt = 1.0):
        self.time_vector = np.arange(0, tsim, dt)      # Time vector
        self.spike_train = np.zeros(int(tsim / dt))    # Spike Train Vector
        self.S_t = sm.Signal(seed_signal = seed_signal, t_signal = tsim, dt = dt)
        np.random.seed(seed = seed_noise)              # Noise Seed
        # Simulation
        for i in range(int(tsim / dt)):
            r_t = self.r0*(1 + self.epslon*self.S_t.signal[i])
            if  np.random.rand() < r_t*dt*1e-3:
                self.spike_train[i] = 1.0 / dt
        return self.spike_train, self.S_t.signal

class stochsticLIF(object):

    r'''
        Constructor
        > tau_m: Membrane time constant
        > mu:    Rest potential in mV
        > D:     Overall noise intensity
    	> c:     Relative strength of the signal
    	> tabs:  Refractory period in ms
    '''
    def __init__(self, tau_m = 10.0, mu = 0.5, D = 5, c = 0.8, tabs = 2.0):
        self.tau_m = tau_m
        self.mu    = mu
        self.D     = D
        self.c     = c
        self.tabs  = tabs

    r'''
        Model diferential equation
    '''
    def model_dif_equation(self, membrane_potential, signal, noise):
        return ( -membrane_potential + self.mu + np.sqrt(2*self.D*self.c)*signal + np.sqrt(2*self.D*(1-self.c))* noise ) / self.tau_m

    r'''
        Solve model equations
        tsim: Simulation time
        dt:   Time resolution
    '''
    def solve(self, seed_signal = 1, seed_noise = 2, tsim = 1000, dt = 1.0):
        self.time_vector = np.arange(0, tsim, dt)      # Time vector
        self.spike_train = np.zeros(int(tsim / dt))    # Spike Train Vector
        v      = 0.0                                   # Membrane potential vector
        vt, vr     = 1.0, 0.0                          # Threshold, reset
        X = 0
    	tspike = 0
        self.S_t = sm.Signal(seed_signal = seed_signal, t_signal = tsim, dt = dt)
        self.N_t = sm.Signal(seed_signal = seed_noise, t_signal = tsim, dt = dt)
        # Simulation
        for i in range(0, int(tsim / dt)) :
    		if X == 1:
    			v = vr
    			if self.time_vector[i] - tspike >= self.tabs:
    				X = 0
    		else:
    			v = v + dt * self.model_dif_equation(v, self.S_t.signal[i], self.N_t.signal[i])
    			if v > vt:
    				self.spike_train[i] = 1.0 / dt
    				tspike         = self.time_vector[i]
    				v              = vr
        return self.spike_train, self.S_t.signal
