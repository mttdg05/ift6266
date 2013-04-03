
import numpy as np
import utils
import kmeans
import encoder
import pooling

if __name__ == '__main__'  :
  k = 1000
  d = 2304
  d_sqrt = int(np.sqrt(d))
  dpatch_sqrt = 6
  dpatch = dpatch_sqrt ** 2

  # compute the encoder
  alpha = 0.005
  whiten_patches = np.load('data/whiten/train/train_whiten_patches0.npy')
  centers = np.load('data/centers.npy') 
  whiten_patches = whiten_patches[ 1386750 : 1849000 ].copy()
  print 'computing encoder ...'
  features = encoder.kmeans_triangle(whiten_patches, centers)
  print 'done'
  n = int(features.shape[0] / ( ((d_sqrt - dpatch_sqrt + 1) ** 2)))
  features = features.reshape(( n, (d_sqrt - dpatch_sqrt + 1), (d_sqrt - dpatch_sqrt + 1), k))

  # Pool the features -> mean pooling
  pooled_features = pooling.meanPooling(features)
  pooled_features = utils.normalize(pooled_features)
  np.save( "data/pooled_features/train/train_pooled_features3.npy", pooled_features )
  
