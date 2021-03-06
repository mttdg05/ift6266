template = """
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
  whiten_patches = np.load(%(whiten_patches_path)s)
  centers = np.load('data/centers.npy') 
  whiten_patches = whiten_patches[ %(start)i : %(end)i ].copy()
  print 'computing encoder ...'
  features = encoder.kmeans_triangle(whiten_patches, centers)
  print 'done'
  n = int(features.shape[0] / ( ((d_sqrt - dpatch_sqrt + 1) ** 2)))
  features = features.reshape(( n, (d_sqrt - dpatch_sqrt + 1), (d_sqrt - dpatch_sqrt + 1), k))

  # Pool the features -> mean pooling
  pooled_features = pooling.meanPooling(features)
  pooled_features = utils.normalize(pooled_features)
  np.save( %(pooled_features_path)s, pooled_features )
  
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("which_set", help = "which_set : train || test")
args = parser.parse_args()
which_set = str(args.which_set)


if which_set == 'train' :
    whiten_patches_paths = ["'data/whiten/train/train_whiten_patches0.npy'", "'data/whiten/train/train_whiten_patches1.npy'", "'data/whiten/train/train_whiten_patches2.npy'", "'data/whiten/train/train_whiten_patches3.npy'"]
    # the 3 first batches have 1.849.000 points -> split them in 4
    # the last batch have 2.178.122 points split -> split it in 5
    n_batches = [4, 4, 4, 5]
    param_sets = []

    step  = 462250 # 1.849.000 / 4
    start = -step
    end   = 0
    count = 0

    for i, whiten_patches_path in enumerate(whiten_patches_paths) : 
        start = -step
        end   = 0
        for j in xrange(n_batches[i]) :
            start += step
            end   += step
            if j == 4 : end = start + 329122 # last batch
            pooled_features_path = ''.join(['"data/pooled_features/train/train_pooled_features', str(count), '.npy"'])
            param_sets.append({'whiten_patches_path' : whiten_patches_path, 'start' : start, 'end' : end, 'pooled_features_path' : pooled_features_path})
            count += 1
    
    for i, param_set in enumerate(param_sets):
        content = template % param_set
        title = ''.join(['encod_pool/train/encod_pool', str(i), '.py'])
  	f = open(title, 'w')
	f.write(content)
  	f.close()

elif which_set == 'test' :
    whiten_patches_paths = ["'data/whiten/test/test_whiten_patches0.npy'", "'data/whiten/test/test_whiten_patches1.npy'"]
    # the 2 batches have 1.212.944 points -> split them in 4
    n_batches = [4, 4]
    param_sets = []

    step  = 303236 # 1.212.944 / 4
    start = -step
    end   = 0
    count = 0

    for i, whiten_patches_path in enumerate(whiten_patches_paths) : 
        start = -step
        end   = 0
        for j in xrange(n_batches[i]) :
            start += step
            end   += step
            pooled_features_path = ''.join(['"data/pooled_features/test/test_pooled_features', str(count), '.npy"'])
            param_sets.append({'whiten_patches_path' : whiten_patches_path, 'start' : start, 'end' : end, 'pooled_features_path' : pooled_features_path})
            count += 1
    
    for i, param_set in enumerate(param_sets):
        content = template % param_set
        title = ''.join(['encod_pool/test/encod_pool', str(i), '.py'])
  	f = open(title, 'w')
	f.write(content)
  	f.close()

