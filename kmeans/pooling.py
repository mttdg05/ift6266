import numpy as np

def meanPooling(X) :
  '''
  mean pooling over 4 quadrants
  X is (n_data_points, n_patches, n_patches, k)
  '''
  print 'mean pooling ...'
  n_data_points, n_patches, n_patches, k = X.shape
  half = int(round(n_patches / 2))
  pool = np.zeros((n_data_points, 2, 2, k))
  # uhhhh -> find a cleaner way
  pool0 = np.mean(np.mean(X[:, 0:half, 0:half], axis = 1), axis = 1)
  pool1 = np.mean(np.mean(X[:, 0:half, half:], axis = 1), axis = 1)
  pool2 = np.mean(np.mean(X[:, half:, 0:half], axis = 1), axis = 1)
  pool3 = np.mean(np.mean(X[:, half:, half:], axis = 1), axis = 1)
  print 'done pooling'

  # flatten the pooled features in a vector
  return np.hstack((pool0, pool1, pool2, pool3))
  

def maxPooling(X)  :
  pass
