import numpy as np
import utils
import kmeans


if __name__ == '__main__'  :
  train_inputs = np.load('data/raw/raw_train_inputs.npy')
  n, d = train_inputs.shape 
  d_sqrt = int(np.sqrt(d))
  dpatch_sqrt = 6
  dpatch = dpatch_sqrt ** 2
   
  
  # extract random patches 
  # kmeans requires lots of patches 
  # so i take ~ 400.000 patches
  crop_n_patches = 100
  R = utils.random_patch(d_sqrt, dpatch_sqrt, crop_n_patches)
  X = train_inputs[:, R].reshape((n * crop_n_patches, dpatch))
  X.astype('float32')
  
  # preprocessing
  print 'preprocessing random patches ...'
  whitening_matrix = utils.compute_whitening_matrix(X)
  inv_whitening_matrix = np.linalg.inv(whitening_matrix)
  whiten_X = np.dot(whitening_matrix, X.T).T
  print 'done preprocessing'
  
  k = 1000
  iterations = 20
  centers = kmeans.spherical_kmeans(whiten_X, k, iterations)
  #utils.show_images(centers.reshape(100, dpatch_sqrt, dpatch_sqrt))
  np.save('data/centers', centers)
 
  
 
  
