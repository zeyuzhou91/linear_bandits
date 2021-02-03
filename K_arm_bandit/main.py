import numpy as np
import matplotlib.pyplot as plt
import auxiliary as aux
from copy import copy
import Thompson_Sampling as TS
import Particle_Thompson_Sampling as PTS
import Particle_Regeneration1 as PR1



np.set_printoptions(precision=4)


def run_simulations(K, T, Npar, T_ephch, N_simul, alg):
    
    x = np.zeros(T)
    for i in range(N_simul):
        if i % 1 == 0:
            print('Simulation ', i)
            
        if alg == 'PR1':
            G = PR1.System_PR1(K,T,Npar,T_epoch)
            G.init_true_parameter()
            G.find_best_action()
            G.init_particles()
            G.run()
            AVG_REG += G.AVG_REG
        
        if alg == 'TS':
            G = TS.System_TS(K,T)
            G.init_true_parameter()
            G.find_best_action()
            G.run()
            x += G.CUM_REG
    
    print('theta_true =', G.theta_true)  
    print('Estimate of theta_1 =', G.Alpha[0]/(G.Alpha[0]+G.Beta[0]))
    print('Estimate of theta_2 =', G.Alpha[1]/(G.Alpha[1]+G.Beta[1]))
    x = x / N_simul
    
    return x
        

       
if __name__ == "__main__":
    
    # Set up model parameters
    K = 2       # number of arms
    T = 500      # time horizon
    Npar = 3    # number of particles
    T_epoch = 4  # Epoch time 
    N_simul = 1   # number of simulations
    
    
    CUM_REG = run_simulations(K,T,Npar,T_epoch,N_simul,'TS')
    
    # plot   
    
    plt.figure(2) 
    plt.plot(range(T), CUM_REG, label='TS')
    plt.legend()
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('cumulative regret')
    plt.title(str(K) + '-arm Bernoulli bandit')
    plt.show()