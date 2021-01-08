import numpy as np
import scipy as sp
import scipy.stats as st


def argmax_of_array(array):
    """
    Find the index of the largest value in an array of real numbers. 
    
    Input:
      array:  an array of real numbers. 
    
    Output:
      index:  an integer in [K], where K = len(array)
    """
    
    # Simple but does not support random selection in the case of more than one largest values. 
    ind = np.argmax(array)
    return ind


def argmax_multi_domain_with_context(G, theta_hat, c):
    """
    Find the best action in the multi-domain network slicing problem, given context c
    and assuming the system parameter is theta_hat. 
    
    Input:
      theta_hat:  a numpy array of dimension (D, max(B), 3).
      c:          the context, a vector of length 3. 
    
    Output:
      a:          an action, a length-D vector of integers
    """
    
    a = np.zeros(G.D)
    for i in range(G.D):
        #print('shape of c: ', np.shape(c))
        #print('theta_hat[i]: ', theta_hat[i][:G.B[i]])
        #print('shape of theta_hat[i]: ', np.shape(theta_hat[i][:G.B[i]]))
        mu = theta_hat[i][:G.B[i]].dot(c)   # the vector of mu's in domain i
        #print('mu =', mu)
        a[i] = int(np.argmax(mu))
        #print('a[i] =', a[i])
    
    return a