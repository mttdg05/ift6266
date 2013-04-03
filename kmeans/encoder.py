import numpy as np

#  max{0, X * D.T - alpha}
def soft_threshold(X, D, alpha) :
  '''
  X is n x d
  D is k x d
  alpha is a scalar constant
  '''
  proj = np.dot(X, D.T) - alpha
  proj[ proj < 0 ] = 0.0
  return proj




def kmeans_triangle(X, C) : 
  # data is nxd
  X2 = (X ** 2).sum(1)[:, None]
  C2 = (C ** 2).sum(1)[:, None]
  D = -2 * np.dot(C, X.T) + C2 + X2.T
  mu = np.mean(D, axis = 1)[:, None]
  f = mu - D
  f[f < 0.0] = 0.0

  return f.T # nxk

'''
n, d, k = 5,4,3
x = np.arange(0, n*d).reshape((n, d))
y = np.arange(0, k*d).reshape((k, d))
print kmeans_triangle(x, y).shape
print soft_threshold(x, y, 0.5).shape
'''

