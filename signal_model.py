import numpy as np

r'''
    Class Signal, to generate stimulus or noise drawn for a normalized gaussian
    distribution.
'''
class Signal(object):

    r'''
        Constructor
        > seed_signal: Seed to generate signal
        > t_signal   : Signal time duration
        > dt         : Signal resolutian (default 1ms)
        > gen        : If true call generator methods
    '''
    def __init__(self, seed_signal = 1000, t_signal = 1000, dt = 1.0, gen = True):
        self.seed_signal = seed_signal
        self.dt          = dt
        self.t_signal    = t_signal
        if gen == True:
            self.gen_time_vector()
            self.gen_signal()

    r'''
        Generate the time vector
    '''
    def gen_time_vector(self, ):
        self.time_vector = np.arange(0, self.t_signal, self.dt)

    r'''
        Generate the signal
    '''
    def gen_signal(self, ):
        np.random.seed(seed = self.seed_signal)
        self.signal      = np.random.randn(int(self.t_signal / self.dt))
