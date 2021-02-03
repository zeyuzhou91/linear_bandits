import numpy as np
import scipy as sp
import scipy.stats as st
import Game
import auxiliary as aux


class System_PTS(Game.System): 
    def __init__(self, K, T, Npar):
        Game.System.__init__(self, K, T) 
        self.Npar = Npar                        # number of particles
        self.Particles = np.zeros(Npar)    # the set of particles, un-initialized
        self.w = np.ones(Npar) * (1.0/Npar)     # the weights of the particles, initialized to be all having equal weights
        self.w_bar = np.ones(Npar) * (1.0/Npar) # the running average weights of the particles 
        self.w_history = np.zeros((T, Npar))    # history of the weights
        self.w_bar_history = np.zeros((T, Npar)) # history of the runninv average weights
    
    
    def init_particles(self):
        """
        Initialize the set of particles. 
        """
        
        # Method 1: Each particle is a dimension-K vector. We generate each particle 
        # uniformly at random from the space [0,1]^K. 
        # This method for any integer K.
        self.Particles = np.random.uniform(0, 1, self.Npar)
        #print("Particles: ", self.Particles)
        
        
        # Method 2: We generate m points on [0,1] uniformly at random and let the set
        # of particles be the K-fold meshgrid of these m points. 
        # E.g. If m = 3 and the points are [0.1, 0.4, 0.7]. 
        # Then for K = 2, the particles are [0.1, 0.1], [0.1, 0.4], [0.1, 0.7],
        # [0.4, 0.1], [0.4, 0.4], [0.4, 0.7], [0.7, 0.1], [0.7, 0.4], [0.7, 0.7].
        # This method requires Npar = m^K.
        
    
        # Method 3: Pre-determined points. 
        ## Case 1:
        #Par = np.array([[0.72, 0.4],
                        #[0.8, 0.59],
                        #[0.1, 0.2],
                        #[0.4, 0.9],
                        #[0.5, 0.3]])
        
        ## Case 2:
        #Par = np.array([[0.72, 0.9],
                        #[0.3, 0.59],
                        #[0.1, 0.2],
                        #[0.4, 0.9],
                        #[0.5, 0.3]]) 
        
        ## Case 3:
        #Par = np.array([[0.72, 0.4],
                        #[0.3, 0.59],
                        #[0.1, 0.2],
                        #[0.4, 0.9],
                        #[0.5, 0.3]]) 
        
        ## Case 4:
        #Par = np.array([[0.72, 0.9],
                        #[0.8, 0.59],
                        #[0.1, 0.2],
                        #[0.4, 0.9],
                        #[0.5, 0.3]])    
        #print("The set of particles is: ", Par)
        
        
        ### Case 5: (3 particles)
        #thetatrue1 = 0.7
        #thetatrue2 = 0.5
        #d11 = 0.7
        #d12 = 0.05
        #d21 = 0.1
        #d22 = 0.8
        #d31 = 0.25
        #d32 = 0.25
        
        #epsilon = 0.00000001
        
        ## particle 1
        #(theta11_1, theta11_2) = aux.KL_inverse(thetatrue1, d11, epsilon, thetatrue1/2.0, thetatrue1 + (1-thetatrue1)/2.0)
        #(theta12_1, theta12_2) = aux.KL_inverse(thetatrue2, d12, epsilon, thetatrue2/2.0, thetatrue2 + (1-thetatrue2)/2.0)
        #theta1 = np.array([theta11_1, theta12_1])
        
        ## particle 2
        #(theta21_1, theta21_2) = aux.KL_inverse(thetatrue1, d21, epsilon, thetatrue1/2.0, thetatrue1 + (1-thetatrue1)/2.0)
        #(theta22_1, theta22_2) = aux.KL_inverse(thetatrue2, d22, epsilon, thetatrue2/2.0, thetatrue2 + (1-thetatrue2)/2.0) 
        #theta2 = np.array([theta21_1, theta22_2])
        
        ## particle 3
        #(theta31_1, theta31_2) = aux.KL_inverse(thetatrue1, d31, epsilon, thetatrue1/2.0, thetatrue1 + (1-thetatrue1)/2.0)
        #(theta32_1, theta32_2) = aux.KL_inverse(thetatrue2, d32, epsilon, thetatrue2/2.0, thetatrue2 + (1-thetatrue2)/2.0)  
        #theta3 = np.array([theta31_1, theta32_1])
            
        #Par[0] = theta1
        #Par[1] = theta2
        #Par[2] = theta3
        ##print("The set of particles is: ", Par)
                
        return None    
    
    
    def select_action(self, t):
        """
        Use Particle Thompson Sampling to select an action.
          
        Input:
          t:    the round index, 0 <= t <= T-1.
          
        Output:
          a:   an action/arm, an integer in [K]. 
        """
        
        theta_hat = generate_parameter_sample(self)
        #print('theta_hat:', theta_hat)
        
        #a = int(aux.argmax_of_array(theta_hat))
        a = 0
        #print('Actual Action:', a)
        
        return a 
    
    
    def update_state(self, a, obs, t):
        """
        Update the state variables given action a and observation obs. 
        
        Input:
          a:    the action taken in round t, an integer in [K]
          obs:  the observation incurred in round t, 0 or 1
          t:    the round index, 0 <= t <= T-1. 
        """
        
        self.update_weights(a, obs, t)   # only update weights, not particles   
        self.update_running_average_weights(t) 
        return None    


    def update_weights(self, a, obs, t):
        """
        Update the weights of the particles.  
        
        Input:
          a:    the action/arm taken in round t, an integer. 
          obs:  the observation incurred in round t, 0 or 1.
          t:    the round index, 0 <= t <= T-1. 
        """
        
        new_w_tilde = np.zeros(self.Npar)  # the unnormalized new weight vector
        for k in range(self.Npar):
            lh = calculate_likelihood(self.Particles[k], obs)
            #print('likelihood =', lh)
            new_w_tilde[k] = lh * self.w[k]
        new_w = 1.0/(np.sum(new_w_tilde)) * new_w_tilde   # normalizing
        #print('new_w =', new_w)
        self.w = new_w
        
        #print('Particle weigths =', self.w)
        
        self.update_w_history(self.w, t)
        return None    
    
    
    def update_running_average_weights(self, t):
        """
        Update the running average weights of the particles.  
        
        Input:
          t:    the round index, 0 <= t <= T-1. 
        """    
        
        self.w_bar = self.w_bar * (float(t)/(t+1)) + self.w / float(t+1) 
        #print('Running average particle weigths =', self.w_bar)
        
        self.update_w_bar_history(self.w_bar, t)
        return None


    def update_w_history(self, w, t):
        """        
        Input:
          w:      the weights of the particles in round t, a probability vector of length Npar. 
          w_bar:  the running average weights of the particles in round t, a probability vector of length Npar.
          t:      the round index, 0 <= t <= T-1. 
        """        
        self.w_history[t, :] = w
        return None    


    def update_w_bar_history(self, w_bar, t):
        """        
        Input:
          w_bar:  the running average weights of the particles in round t, a probability vector of length Npar.
          t:      the round index, 0 <= t <= T-1. 
        """        
        self.w_bar_history[t, :] = w_bar
        return None  
    
    
    def print_state(self):
        """
        Print the current value of the state variables."
        """
        
        print('Particles = ', self.Particles)
        print('Weights = ', self.w_history)  
        return None


def generate_parameter_sample(G):
    """
    Generate a sample theta_hat (one particle) based on the current weights on the particles. 
    
    Input:
      G:   a game system object. 
    
    Output:
      theta_hat: a length-K vector of values in [0,1].
    """   
    
    theta_hat = np.zeros(G.K) 
    k = np.random.choice(G.Npar, 1, p=G.w)[0]  # np.random.choice outputs an array
    theta_hat = G.Particles[k]   
    # ZEYU: is there a quicker way to implement this?
    return theta_hat



def calculate_likelihood(theta, obs):
    """
    Calculate the likelihood/probability of observing obs, if the parameter is theta. 
    
    Input:
      theta:  a probability. 
      obs:    0 or 1. 
    
    Output:
      lh:     a number in [0,1], the likelihood/probability. 
    """
    
    if obs == 1:
        lh = theta
    else:
        lh = 1-theta
    
    return lh






if __name__ == "__main__":
    pass