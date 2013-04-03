
template = """
import numpy as np
import utils

if __name__ == '__main__'  :
  start = %(start)i
  end = %(end)i
  train_inputs = np.load('data/raw/raw_train_inputs.npy')[start:end]
  n, d = train_inputs.shape
  d_sqrt = int(np.sqrt(d))
  dpatch_sqrt = %(dpatch_sqrt)i
  dpatch = dpatch_sqrt ** 2
   
  # extract training features
  stride = %(stride)i
  patches_indexes = utils.extract_patches(d_sqrt, dpatch_sqrt, stride)
  patches = train_inputs[:, patches_indexes] 
  patches = patches.reshape(( n * (d_sqrt - dpatch_sqrt + 1) ** 2 , dpatch))
  
  if %(do_whitening)i : 
      print 'preprocessing overlapping  patches ...'
      whitening_matrix = utils.compute_whitening_matrix(patches)
      inv_whitening_matrix = np.linalg.inv(whitening_matrix)
      whiten_patches = np.dot(whitening_matrix, patches.T).T
      np.save( %(filepath)s, whiten_patches)
      print 'done preprocessing'
  
  else : 
      np.save( %(filepath)s, patches)
  
"""
import argparse

if __name__ == '__main__'  :
    parser = argparse.ArgumentParser()
    parser.add_argument("which_set", help = "which_set : train || test")
    args = parser.parse_args()
    which_set = str(args.which_set)
    if which_set == 'train' :
        # the train set have 4178 ex -> split it in 4 chunks of 3 * 1000 + 1178
        start = [0, 1000, 2000, 3000]
    	end = [1000, 2000, 3000, 4178]
    	dpatch_sqrt  = [6, 6, 6, 6]
    	stride = [1, 1, 1, 1]
    	do_whitening = [1, 1, 1, 1] # 0 -> False 1 -> True
    	filepath = ['"data/whiten/train/train_whiten_patches0"', '"data/whiten/train/train_whiten_patches1"', '"data/whiten/train/train_whiten_patches2"', '"data/whiten/train/train_whiten_patches3"']
    	param_sets = []
    	for i in xrange(len(start)) :
            param_sets.append({'start' : start[i], 'end' : end[i], 'dpatch_sqrt' : dpatch_sqrt[i], 'stride' : stride[i], 'do_whitening' : do_whitening[i], 'filepath' : filepath[i]})
    
    	for i, param_set in enumerate(param_sets):
            content = template % param_set
            title = ''.join(['patches/train/make_train_batch', str(i), '.py'])
            f = open(title, 'w')
            f.write(content)
            f.close()
     
    elif which_set == 'test': 
        # the test set have 1312 ex -> split it in half
        start = [0, 656]
    	end = [656, 1312]
    	dpatch_sqrt  = [6, 6]
    	stride = [1, 1]
    	do_whitening = [1, 1] # 0 -> False 1 -> True
    	filepath = ['"data/whiten/test/test_whiten_patches0"', '"data/whiten/test/test_whiten_patches1"']
    	param_sets = []
    	for i in xrange(len(start)) :
            param_sets.append({'start' : start[i], 'end' : end[i], 'dpatch_sqrt' : dpatch_sqrt[i], 'stride' : stride[i], 'do_whitening' : do_whitening[i], 'filepath' : filepath[i]})
    
    	for i, param_set in enumerate(param_sets):
            content = template % param_set
            title = ''.join(['patches/test/make_test_batch', str(i), '.py'])
            f = open(title, 'w')
            f.write(content)
            f.close()
    else : 
        raise ValueError("The set must be train or test!")

 
