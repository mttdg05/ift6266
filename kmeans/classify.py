import numpy as np
import math
import random
import pickle
import csv
import utils
import kmeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

if __name__ == '__main__'  :    
  # train an one against all SVM 
  n, d = 4178, 4000
  train_features = np.load('data/features/train_features.npy')
  #train_features = np.load('data/less/whiten_train_features.npy')
  test_features = np.load('data/features/test_features.npy')
  train_labels = np.load('data/train_labels.npy')
  
  indexes = random.shuffle(range(n))
  train_features = (train_features[indexes]).reshape((n, d))
  train_labels =   (train_labels[indexes]).reshape((n,))

  valid_set_size = 0
  x_train = train_features[valid_set_size: ]
  x_valid = train_features[:valid_set_size]
  y_train = train_labels[valid_set_size:]
  y_valid = train_labels[:valid_set_size]


  print 'training an one against all SVM '
  _lambda = 0.05
  model = OneVsRestClassifier(LinearSVC(C = _lambda)).fit(x_train, y_train)
  train_pred = model.predict(x_train)
  print  'incorrect : ', y_train[y_train != train_pred].shape[0] 
  print 'train error : ',   (1.0 * y_train[y_train != train_pred].shape[0] / y_train.shape[0] ) * 100

  valid_pred = model.predict(x_valid)
  print 'valid error : ',   (1.0 * y_valid[y_valid != valid_pred].shape[0] / y_valid.shape[0] ) * 100


  
  test_pred = model.predict(test_features)
  out = open('sub_kmeans0.csv', 'w')
  for i in xrange(test_pred.shape[0]) :
     out.write('%d\n' % test_pred[i]) 
  out.close()
  
  
  
