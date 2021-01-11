import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Test for plotting the contours of a Gaussian distribution

x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x, y))
#print('pos =', pos)
#print('np.shape(pos) =', np.shape(pos))
#print(pos[:,:,0])
#print(pos[:,:,1])
mu = np.array([0.5, -0.2])
Sigma = np.array([[2.0, 0.3], [0.3, 0.5]])
rv = st.multivariate_normal(mu, Sigma)
plt.figure()
plt.contourf(x, y, rv.pdf(pos), cmap='gray', alpha=0.5)