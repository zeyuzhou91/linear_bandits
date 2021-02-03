import numpy as np
import matplotlib.pyplot as plt
import auxiliary as aux
from copy import copy
import Thompson_Sampling as TS
import Particle_Thompson_Sampling as PTS
import Particle_Regeneration1 as PR1



np.set_printoptions(precision=4)


def run_simulations(K, T, Npar, T_ephch, N_simul, alg):
    
    DIST = np.zeros(T)
    for i in range(N_simul):
        if i % 1 == 0:
            print('Simulation ', i)
            
        if alg == 'PR1':
            G = PR1.System_PR1(K,T,Npar,T_epoch)
            G.init_true_parameter()
            G.init_particles()
            G.run()
            DIST += G.dist_history
        
    DIST = DIST / N_simul
    
    return DIST
        

       
if __name__ == "__main__":
    
    # Set up model parameters
    K = 1       # number of arms
    T = 2000      # time horizon
    Npar = 10    # number of particles
    T_epoch = 100  # Epoch time 
    N_simul = 1   # number of simulations
    
    
    DIST_PR1 = run_simulations(K,T,Npar,T_epoch,N_simul,'PR1')
    
    # plot   
    
    plt.figure(2) 
    plt.plot(range(T), DIST_PR1)
    plt.legend()
    plt.grid()
    plt.xlabel('t')
    plt.ylabel(r'$|\mathbb{E}_{w_t}[\theta] - \theta^*|$')
    plt.title('')
    plt.show()