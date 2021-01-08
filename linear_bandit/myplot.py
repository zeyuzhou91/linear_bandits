import numpy as np
import matplotlib.pyplot as plt
import auxiliary as aux


colors = ['green', 'blue', 'orange', 'purple', 'gray', 'yellow']

def plot_figures(G, t):
    """
    Plot figures of the game G at time t.
    
    t:    the round index, 0 <= t <= T-1. 
    """
    
    fig1 = plt.figure(1)
    plt.clf()  # clear the current figure
    plt.ion()  # without this, plt.show() will block the code execution
    fig1.suptitle('time =' + str(t))
        
    ax1 = plt.subplot(131)  # position graph
    plot_position_graph(G, t, ax1)
    
    ax2 = plt.subplot(132)  # cumulative regret
    plot_cumulative_regret(G, t, ax2)

    ax3 = plt.subplot(133)  # fraction of time of pulling arm 1
    plot_arm1_frac_graph(G, t, ax3)
    
    plt.show()
    plt.pause(0.0001) 
    
    if t % G.T_settle == 0:
        fig2 = plt.figure(2)
        plt.clf()  # clear the current figure
        ax = fig2.add_subplot(111)
        plot_divergence_graph(G, t, ax)

    return None


def plot_position_graph(G, t, ax):
    
    ax.scatter(G.theta_true[0], G.theta_true[1], c='red', s=100, alpha=1.0, label='true theta')
    ax.plot(np.linspace(0,1,10), np.linspace(0,1,10), c='black', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')        
    ax.grid()     
    
    for i in range(G.Npar):
        #ax.scatter(G.Particles[i][0], G.Particles[i][1], c=colors[i], s=aux.h(G.W[i])*100, alpha=0.8)
        ax.scatter(G.Particles[i][0], G.Particles[i][1], s=aux.h(G.W[i])*100, alpha=0.8)
        if G.W[i] > 0.001:
            ax.text(G.Particles[i][0]-0.1, G.Particles[i][1]+0.02, '{:.3f}'.format(G.W[i]))    
            
    return None


def plot_cumulative_regret(G, t, ax):
    
    ax.plot(range(t), G.CUM_REG[:t])
    ax.set_xlim([-1, G.T+1])
    #ax.set_ylim([0.0, T/4.0])        
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('cumulative regret')         
    ax.grid()    
    
    return None



def plot_arm1_frac_graph(G, t, ax):

    ax.plot(range(t), G.arm1_frac[:t])
    ax.set_xlim([-1, G.T+1])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$r(t)$')
    ax.grid()
    return None



def plot_divergence_graph(G, t, ax):
    D = aux.calculate_KL_divergence_vectors(G.theta_true, G.Particles)
    
    for i in range(G.Npar):
        ax.plot([0, 1], [D[i][1], D[i][0]])
        if G.Particles[i][0] >= G.Particles[i][1]:
            ax.scatter([1], [D[i][0]], s=50)
        else:
            ax.scatter([0], [D[i][1]], s=50)          
    plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    plt.xlabel('r')
    plt.text(1, -0.02, 'arm 1')
    plt.text(0, -0.02, 'arm 2')
    plt.title('Simulation of the divergence graph')
    plt.grid()    

    r_list = np.linspace(0,1,10001)
    left_r_list = np.array([])
    right_r_list = np.array([])
    for r in r_list:
        eff_D = aux.calculate_effective_divergence(D, r)
        low_idx = np.argmin(eff_D)
        if G.Particles[low_idx][0] >= G.Particles[low_idx][1]:
            right_r_list = np.append(right_r_list, [r])
        else:
            left_r_list = np.append(left_r_list, [r])
            
    plot_drift_graph(left_r_list, right_r_list, ax)  
    return None    



def plot_drift_graph(l_list, r_list, ax):
    
    l_len = len(l_list)
    r_len = len(r_list)
    ax.scatter(l_list, (-0.5)*np.ones(l_len), color='red', s=5, label='drift to left')
    ax.scatter(r_list, (-0.5)*np.ones(r_len), color='blue', s=5, label='drift to right')
    ax.legend()
    
    return None




