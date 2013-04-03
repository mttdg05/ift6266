import numpy as np
import math

def spherical_kmeans(X, k, iterations):
  n, d = X.shape
  n_range = np.array(range(n))
  D = np.random.normal(size = (k, d))
  for i in xrange(k) : # unit lenght normalization
      D[i] *= 1.0 / math.sqrt(np.dot(D[i], D[i]))


  for iter in xrange(iterations) :
    print 'Iter # ', iter  
    dot_prods = np.dot(X, D.T)
    winner = np.argmax(np.abs(dot_prods), axis = 1)
    s = dot_prods[(n_range, winner)]
    for i in xrange(k) :
      pos = (winner == i)
      # non empty cluster
      if X[pos].shape[0] != 0 : 
        D[i, :] += np.dot(s[pos], X[pos])
      # if we have an empty cluster -> reinitialize the centroid with a random point
      else :
        D[i, :] =  X[np.random.randint(0, n)]
        
      # unit lenght normalization
      D[i] *= 1.0 / math.sqrt(np.dot(D[i], D[i]))
 
  return D


def hard_kmeans(X, K, initial_centers = None, niterations = 20) :
  n, d = X.shape

  # Initialize centers (if initial_centers not specified, pick at random from the dataset X)
  if initial_centers is not None :
    centers = initial_centers.copy()
  else:
    centers = np.zeros((K, d))
    indexes = []
    while len(indexes) < K :
      i = np.random.randint(0,n)
      if i not in indexes:
        indexes.append(i)
    centers = X[indexes]
    #show_images(centers)
                
  # Do a number of k-means iterations
  for i in range(niterations) :
    print 'Iter # ', i
    sqdists = sqdistances(X, centers)
    winner = np.argmin(sqdists, axis = 1)
    for k in range(K) :
      # non empty cluster
      if X[winner == k].shape[0] != 0 : 
        centers[k, :] = X[winner == k].mean(axis = 0)
      # empty cluster -> take a random point
      else :
        centers[k, :] = X[np.random.randint(0, n)]
        
  return centers



def sqdistances(X, centers):
    """Computes the squared Euclidean distances between the n1 rows of X and the n2 rows of centers.
    Result is returned as a (n1,n2) ndarray of squared distances"""
    
    n,d = X.shape
    dotprods = np.dot(X,centers.T)
    centers_sqnorm = (centers*centers).sum(axis=1)
    X_sqnorm = (X*X).sum(axis=1)
    sqdists = dotprods
    sqdists *= -2.
    sqdists += centers_sqnorm
    sqdists += X_sqnorm.reshape((n,1))
    return sqdists



'''
def kmeans(X, k, iterations):
  n, d = X.shape
  S = np.zeros((k, n))
  D = np.random.normal(size = (k, d))
  # TO DO normalize to unit length vectors
  for iter in xrange(iterations) :
    for i in xrange(n) :
      for j in xrange(k) :
        if j == argmax( D * X[i], axis = 0) :
          S[j][i] = D[j] * x[i]
        else : 
          S[j][i] = 0.0
        D += np.dot(S, X)
        D[j] /= np.linalg.norm(D[j])

'''
