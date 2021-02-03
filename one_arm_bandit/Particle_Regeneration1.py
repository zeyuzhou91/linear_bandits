import numpy as np
import scipy as sp
import scipy.stats as st
import Particle_Thompson_Sampling as PTS
import auxiliary as aux

class System_PR1(PTS.System_PTS): 
    def __init__(self, K, T, Npar, T_epoch):
        PTS.System_PTS.__init__(self, K, T, Npar) 
        self.T_epoch = T_epoch        # the epoch time
        self.mu = 0                   # the (weighted) mean of the current particles
        self.var = 0                  # the (weighted) variance of the current particles
        self.dist = 0.0                 # A metric that measures the distance between the algorithm's current state and theta_true
                                        # Here, the current state is chosen to be self.mu, the mean value of the current particles
        self.dist_history = np.zeros(T)  # the history of the dist value
    
    def update_state(self, a, obs, t):
        """
        Update the state variables given action a and observation obs. 
        
        Input:
          a:    the action taken in round t, an integer in [K]
          obs:  the observation incurred in round t, 0 or 1
          t:    the round index, 0 <= t <= T-1. (not used here)
        """
        
        self.update_weights(a, obs, t)   # only update weights, not particles 
        
        # Calculate the current weighted mean and variance of the particles
        self.mu = float(self.Particles.dot(self.w))   # E_w[theta]
        #print('np.shape(mu) =', np.shape(mu))
        #print('mu =', self.mu)         
        
        if t > 0 and t % self.T_epoch == 0:
            print('Particle regeneration: ', t / self.T_epoch)
            self.regenerate_particles(t) 
            
        self.update_running_average_weights(t) 
        
        ## Update self.mu and self.var
        ## Note that two options are possible here. We can update these two variables
        ## based on the actual weights self.w or the running average weights self.w_bar.
        ## Test both. 
        
        # Calculate the current weighted mean and variance of the particles
        self.mu = float(self.Particles.dot(self.w))   # E_w[theta]
        #print('np.shape(mu) =', np.shape(mu))
        #print('mu =', self.mu) 
        
        self.dist = abs(self.theta_true - self.mu)
        self.dist_history[t] = self.dist
        return None
    

    
    def regenerate_particles(self, t):
        """
        Regenerate the particles. Calculate the current empirical mean and variance of the particles, 
        then regenerate Npar i.i.d. particles according to the N(mean, variance) distribution. 
        """    
        
        mean_w_square = float((self.Particles**2).dot(self.w))    # E_w[theta^2]
        #print('np.shape(mean_w_square) =', np.shape(mean_w_square))
        #print('mean_w_square =', mean_w_square)        
        self.var = mean_w_square - self.mu**2               # Var_w(theta)
        epo = np.floor(t/self.T_epoch)
        self.var += 0.1/(epo+1)    # TO CONSIDER AND UPDATE
        #print('np.shape(var) =', np.shape(var))
        print('var =', self.var)         
        
        new_Particles = np.random.normal(self.mu, self.var, self.Npar)
        new_Particles = aux.map_to_domain(new_Particles)  # if a particle is outside of [0,1], it should be mapped to 0 or 1
        
        self.Particles = new_Particles
        #print('new particles =', self.Particles)
        self.w = np.ones(self.Npar) * (1.0/self.Npar)
        #print('new weights =', self.w)
        
        return None    
    
    
    def update_running_average_weights(self, t):
        """
        Update the running average weights of the particles.  
        
        Input:
          t:    the round index, 0 <= t <= T-1. 
        """    
        
        # At an epoch, the running average weights need to be updated according to the current time 
        # counting from the most recent epoch, not from the beginning of time 
        t1 = t % self.T_epoch
        if t1 == 0:
            self.w_bar = self.w
        else:
            self.w_bar = self.w_bar * (float(t1)/(t1+1)) + self.w / float(t1+1) 
        #print('Running average particle weigths =', self.w_bar)
        
        self.update_w_bar_history(self.w_bar, t)
        return None    