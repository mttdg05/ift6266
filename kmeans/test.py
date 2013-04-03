import numpy as np
import math
import pickle
import csv
import utils
import kmeans

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


# soft_threshold : max{0, X * D.T - alpha}
def compute_encoder(X, D, alpha) :
  '''
  X is n x d
  D is k x d
  alpha is a scalar constant
  '''
  proj = np.dot(X, D.T) - alpha
  proj[ proj < 0 ] = 0.0
  return proj

def sumPooling(X) :
  '''
  Sum pooling over 4 quadrants
  X is (n_data_points, n_patches, n_patches, k)
  '''
  print 'Sum pooling ...'
  n_data_points, n_patches, n_patches, k = X.shape
  half = int(round(n_patches / 2))
  pool = np.zeros((n_data_points, 2, 2, k))
  # uhhhh -> find a cleaner way
  pool0 = np.sum(np.sum(X[:, 0:half, 0:half], axis = 1), axis = 1)
  pool1 = np.sum(np.sum(X[:, 0:half, half:], axis = 1), axis = 1)
  pool2 = np.sum(np.sum(X[:, half:, 0:half], axis = 1), axis = 1)
  pool3 = np.sum(np.sum(X[:, half:, half:], axis = 1), axis = 1)
  print 'done pooling'

  # flatten the pooled features in a vector
  return np.hstack((pool0, pool1, pool2, pool3))
  
  

if __name__ == '__main__'  :
  #train_inputs = utils.load_data(utils.find_data_file('contest_dataset/train_inputs.p'))
  train_inputs = np.load('train_inputs.npy')
  #np.save('train_inputs', train_inputs)
  #train_labels = utils.load_data(utils.find_data_file('contest_dataset/train_labels.p'))
  #np.save('train_labels', train_labels)

  n, d = train_inputs.shape 
  #n, d = 1000, 2304
  n =  1178
  n1 = 3000#00#0
  n2 = 4178

  train_inputs = train_inputs[n1:n2]
  
  #train_labels = np.array(train_labels[:n]).astype('int32')
  k = 1000
  d_sqrt = int(np.sqrt(d))
  dpatch_sqrt = 6#16
  dpatch = dpatch_sqrt ** 2
   
  '''
  # extract random patches 
  # kmeans requires lots of patches 
  # so i take ~ 400.000 patches
  crop_n_patches = 100#00 
  R = utils.random_patch(d_sqrt, dpatch_sqrt, crop_n_patches)
  X = train_inputs[:, R].reshape((n * crop_n_patches, dpatch))
  X.astype('float32')
  
  # preprocessing
  print 'preprocessing random patches ...'
  whitening_matrix = utils.compute_whitening_matrix(X)
  inv_whitening_matrix = np.linalg.inv(whitening_matrix)
  whiten_X = np.dot(whitening_matrix, X.T).T
  print 'done preprocessing'
 
  # find centroids
  k = 1000#100#1600
  iterations = 10
  centers = kmeans.spherical_kmeans(whiten_X, k, iterations)
  pickle.dump(centers, open('centers.p', 'wb'))
  
  '''
  # showtime 
  #show_n = 256
  #print np.dot(centers[:show_n], inv_whitening_matrix).shape
  #utils.show_images(np.dot(centers[:show_n], inv_whitening_matrix).reshape((show_n, dpatch_sqrt, dpatch_sqrt)))

  ''' 
  # extract training features
  stride = 1
  patches_indexes = utils.extract_patches(d_sqrt, dpatch_sqrt, stride)
  patches = train_inputs[:, patches_indexes] 
  patches = patches.reshape(( n * (d_sqrt - dpatch_sqrt + 1) ** 2 , dpatch))
  print 'preprocessing overlapping  patches ...'
  whitening_matrix = utils.compute_whitening_matrix(patches)
  #inv_whitening_matrix = np.linalg.inv(whitening_matrix)
  whiten_patches = np.dot(whitening_matrix, patches.T).T
  np.save('whiten_patches4', whiten_patches)

  #pickle.dump( whiten_patches, open('whiten_patches4.p', 'wb'))
  print 'done preprocessing'
  '''
  
  # compute the encoder
  alpha = 0.005
  whiten_patches = np.load('whiten_patches1.npy')
  #np.save('whiten_patches1', whiten_patches)
  #centers = utils.load_data('/home/a1/ift6266/kmeans/centers.p')
  centers = np.load('centers.npy')
  
  #np.save('centers', centers)

  n = 250
  #whiten_patches = whiten_patches[924500:1386750]
  whiten_patches = whiten_patches[1386750:]

  features = compute_encoder(whiten_patches, centers, alpha)
  features = features.reshape(( n, (d_sqrt - dpatch_sqrt + 1), (d_sqrt - dpatch_sqrt + 1), k))

  # Pool the features -> sum pooling
  pooled_features = sumPooling(features)
  pooled_features = utils.normalize(pooled_features)
  print pooled_features.shape
  np.save( 'pooled_features4', pooled_features)
  
  '''
  # train an one against all SVM 
  print 'training an one against all SVM '
  model = OneVsRestClassifier(LinearSVC()).fit(pooled_features, train_labels)#.predict(pooled_features)
  train_pred = model.predict(pooled_features)#.astype('int32')
  print 'train error : ',   (1.0 * train_labels[(train_labels) != train_pred].shape[0] / n ) * 100

  
  # test
  test_inputs = utils.load_data(utils.find_data_file('contest_dataset/test_inputs.p'))
  n = test_inputs.shape[0]
  test_inputs = test_inputs[:n]
  #test_labels = np.array(test_labels[:n]).astype('int32')
  
  # extract test features
  #stride = 1
  #patches_indexes = utils.extract_patches(d_sqrt, dpatch_sqrt, stride)
  test_patches = test_inputs[:, patches_indexes] 
  test_patches = test_patches.reshape(( n * (d_sqrt - dpatch_sqrt + 1) ** 2 , dpatch))
  print 'preprocessing overlapping  patches ...'
  whitening_matrix = utils.compute_whitening_matrix(test_patches)
  #inv_whitening_matrix = np.linalg.inv(whitening_matrix)
  whiten_patches = np.dot(whitening_matrix, test_patches.T).T
  print 'done preprocessing'

  
  # compute the encoder
  test_features = compute_encoder(test_patches, centers, alpha)
  test_features = test_features.reshape(( n, (d_sqrt - dpatch_sqrt + 1), (d_sqrt - dpatch_sqrt + 1), k))

  # Pool the features -> sum pooling
  test_pooled_features = sumPooling(test_features)
  test_pooled_features = utils.normalize(test_pooled_features)

  test_pred = model.predict(test_pooled_features)#.astype('int32')
  out = open('sub0.csv', 'w')
  for i in xrange(test_pred.shape[0]) :
    out.write('%d\n' % test_pred[i]) #test_pred
  out.close()
  #print 'test error : ',   (1.0 * test_labels[(test_labels) != test_pred].shape[0] / n ) * 100
  '''




  
  '''
  # extract random patches
  crop_n_patches = 50#00
  R = np.zeros((crop_n_patches, dpatch))
  for k in range(crop_n_patches) :
    R[k] = utils.random_patch(image_height, (dpatch_sqrt, dpatch_sqrt))
  
  X = train_inputs[:, R].reshape((n * crop_n_patches, dpatch))
  X.astype('float32')
  
  # preprocessing
  print 'preprocessing ...'
  whitening_matrix = utils.compute_whitening_matrix(X)
  inv_whitening_matrix = np.linalg.inv(whitening_matrix)
  whiten_X = np.dot(whitening_matrix, X.T).T
  print 'done preprocessing'
 
  #kmeans testing
  print 'kmeans : '
  centers = spherical_kmeans(X, k, iterations)

  test_kmeans(whiten_X, save_centers = 1, inv_whitening_matrix = inv_whitening_matrix)
  #test_kmeans(X)
  '''











