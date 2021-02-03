import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import auxiliary as aux


#colors = ['green', 'blue', 'orange', 'purple', 'gray', 'yellow']



def plot_figures(G, t):
    """
    Plot figures of the game G at time t.
    
    t:    the round index, 0 <= t <= T-1. 
    """
    
    fig1 = plt.figure(1)
    plt.clf()  # clear the current figure
    plt.ion()  # without this, plt.show() will block the code execution

    # Plot the true theta
    plt.stem([G.theta_true], [1], linefmt='red', markerfmt='ro', label=r'$\theta^*$')
    #plt.stem([G.theta_true], [1], linefmt='red', markerfmt='red', label=r'$\theta^*$')
    
    # Plot current particles
    plt.stem(G.Particles, G.w, label='Particles')
    
    # Plot the Gaussian distribution based on the current particles
    #xlist = np.linspace(0,1,100)
    #ylist = st.norm.pdf(xlist, G.mu, np.sqrt(G.var))
    #plt.plot(xlist, ylist,color='black')
    
    plt.xlim([0, 1])
    #plt.ylim([0, 1.5])    
    plt.legend()
    plt.xlabel(r'$\theta$')
    plt.ylabel('weight') 
    plt.title('time =' + str(t))
    plt.grid()     
    
    plt.show()
    plt.pause(0.001) 
    
    return None









