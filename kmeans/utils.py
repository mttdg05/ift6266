import numpy as np
import os
from matplotlib import pyplot as plt



def show_images(images, numrows = None, numcols = None, cmap=plt.cm.gray, flip = 0):
  initPos = 1
  n = images.shape[0]
  if numrows is None or numcols is None or n < numrows * numcols : 
    numrows = int(np.sqrt(n))
    numcols = numrows

  for i in range(numrows * numcols):
    im = images[i]
    axes = plt.subplot(numrows, numcols, initPos+i)
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])
    #axes.set_xlabel(titles[i])
    #plt.imshow(np.flipud(im), cmap=cmap, interpolation='nearest')
    if flip : 
      plt.imshow(np.flipud(im), cmap=cmap)
    else :
      plt.imshow(im, cmap=cmap)


  plt.show()


def extract_patches(d, dpatch, stride) :
  '''
  We have a dxd input image & we extract
  patches of dpatch x dpatch. 
  Returns a n_patch_sqrt x n_patch_sqrt x dpatch * dpatch 
  array containing the indexes of the extracted patches.
  '''
  n_patch_sqrt = d - dpatch + 1
  patches = np.zeros((n_patch_sqrt, n_patch_sqrt, dpatch * dpatch))
  posX, posY = (0, 0)
  while posX + dpatch <= d :
    while posY + dpatch <= d : 
      x = np.arange(posX, posX + dpatch)
      y = range(posY, posY + dpatch)
      patches[posX, posY] = np.array([x + i * d for i in y]).flatten() 
      posY += stride
    posY = 0
    posX += stride
  return patches.astype('int32')


def random_patch(d, dpatch, n):
  # 32x32 images
  #size = 32
  '''
  We have a dxd input image & we extract
  n random patches of dpatch x dpatch. 
  Returns a n x dpatch * dpatch 
  array containing the indexes of the extracted patches.
  '''

  patches = np.zeros((n, dpatch * dpatch))
  startX = np.random.randint(0, d - dpatch)
  startY = np.random.randint(0, d - dpatch)
  flat = np.zeros((d * d))
  not_flat = np.zeros((d, d))
  #print startX, startY
  x = np.arange(startX, startX + dpatch)
  y = range(startY, startY + dpatch)
  for c in xrange(n) :
    patches[c] = np.array([x + i * d for i in y]).flatten()
  return patches.astype('int32')
  #return get_flatten_indexes(flat, not_flat[x][:, y])



def normalize(patches, eps_norm = 10) :
  '''
  mean center & divide by the standard deviation
  '''
  n_patches, d_patch = patches.shape
  d_patch_sqrt = int(np.sqrt(d_patch))
  patches -= patches.mean(1)[:, None]
  patches *= d_patch_sqrt / np.sqrt(np.sum(patches ** 2, axis = 1) + eps_norm)[:, None]
  return patches



def compute_whitening_matrix(patches, eps_norm = 10, eps_zca = 1e-2) :
  '''
  patches is n_patches x d_patch
  '''
  n_patches, d_patch = patches.shape
  d_patch_sqrt = int(np.sqrt(d_patch))
  '''
  for i in xrange(n_patches):
    tmp = patches[i]
    tmp -= np.mean(tmp)
    tmp /= np.sqrt(np.sum(tmp ** 2)) + 0.001
    patches[i] = tmp
  '''
  patches -= patches.mean(1)[:, None]
  patches *= d_patch_sqrt / np.sqrt(np.sum(patches ** 2, axis = 1) + eps_norm)[:, None]

  #patches /= patches.std(1)[:, None] + eps_norm
  #patches /= np.sqrt(np.sum(patches ** 2)) + 0.001

  cov = np.dot(patches.T, patches) / n_patches
  [U, L, V] = np.linalg.svd(cov, compute_uv = 1)
  L = np.diag(L[:]) #+ eps_zca
  #print L
  A = np.dot(U, np.dot(np.sqrt(np.linalg.inv(L)), U.T))
  #A = np.dot(U, np.dot( np.linalg.inv(np.sqrt(L)), U.T)  )

  return A

  #print A.shape
  #print patches.shape 
  #return  np.dot(A, patches.T).T
  #patches = np.dot(A, patches.T).T
  #show_images(patches.reshape((n_patches, d_patch_sqrt, d_patch_sqrt)), 1, 1, cmap=plt.cm.gray)

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
  return D

def init_with_kmeans(X, K, initial_centers=None, niterations=1):
        n,d = X.shape

        # Initialize centers (if initial_centers not specified, pick at random from the dataset X)
        if initial_centers is not None:
            centers = initial_centers.copy()
        else:
            centers = np.zeros((K,d))
            indexes = []
            while len(indexes)<K:
                i = np.random.randint(0,n)
                if i not in indexes:
                    indexes.append(i)
            centers = X[indexes]
                
        # Do a number of k-means iterations
        for i in range(niterations):
            show_images(centers.reshape((30, 5, 5)), 6, 5)
            #print i
            #print centers
            sqdists = sqdistances(X,centers)
            winner = np.argmin(sqdists,axis=1)
            for k in range(K):
                centers[k,:] = X[winner==k].mean(axis = 0)

        return centers
         

def find_data_file(filename):
    home = os.environ['HOME']
    path_starts = [home+"/data/","/data/lisa/data/"]
    for start in path_starts:
        path = start+filename
        if os.path.exists(path):
            return path
    return None

def load_data(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
