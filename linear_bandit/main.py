import numpy as np
import matplotlib.pyplot as plt
import auxiliary as aux
from copy import copy
#import Particle_Thompson_Sampling as PTS
import Thompson_Sampling as TS



np.set_printoptions(precision=4)

       
if __name__ == "__main__":
    
    # Set up model parameters
    N = 2        # dimension of the parameter space
    var_W = 0.1  # variance of the noise W
    T = 500      # time horizon
    
    N_simul = 1   # number of simulations
    
    AVG_REG_TS = np.zeros(T)   # average regret
    
    # run simulations
    for i in range(N_simul):
        if i % 1 == 0:
            print('Simulation', i)
        
        G_TS = TS.System_TS(N, var_W, T)
        G_TS.init_true_parameter()
        G_TS.find_best_action()
        G_TS.run()
        AVG_REG_TS += G_TS.AVG_REG    

        
    AVG_REG_TS = AVG_REG_TS / N_simul


    ## plot   
    
    plt.figure(2) 
    plt.plot(range(T), AVG_REG_TS, label='Thompson sampling')
    plt.legend()
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('running average regret')
    plt.title(r'$N =$' + str(N) + '  ' + r'$\sigma^2_W =$' + str(var_W))
    plt.show()