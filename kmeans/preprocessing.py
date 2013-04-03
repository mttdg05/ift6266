import numpy as np
import math
import pickle
import csv
import utils
import kmeans

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


  
if __name__ == '__main__'  :
  X = np.load('data/features/train_features.npy')
  print X.shape
  print 'preprocessing features...'
  whitening_matrix = utils.compute_whitening_matrix(X)
  inv_whitening_matrix = np.linalg.inv(whitening_matrix)
  whiten_patches = np.dot(whitening_matrix, X.T).T
  np.save('data/features/whiten_train_features', whiten_patches)
  print 'done preprocessing'
  
  
