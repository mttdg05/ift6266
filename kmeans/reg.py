import numpy as np
import math
import random
import pickle
import csv
import utils
import kmeans
import logreg

if __name__ == '__main__'  :    
    # train an one against all SVM 
    
    #train_features = np.load('data/features/train_features.npy')    
    train_features = np.load('data/features/whiten_train_features.npy')  
    test_features = np.load('data/features/test_features.npy')
    train_labels = logreg.onehot(np.load('data/train_labels.npy'))

    n, d, numclasses = 4178, 4000, 7
    indexes = random.shuffle(range(n))
    train_features = (train_features[indexes]).reshape((d, n))
    train_labels   = (train_labels[indexes]).reshape((numclasses, n))
     
    valid_set_size = 700
    x_train = train_features[:, valid_set_size: ]
    x_valid = train_features[:, :valid_set_size]
    y_train = train_labels[:, valid_set_size:]
    y_valid = train_labels[:, :valid_set_size]

    
    '''
    #make some random toy-data:
    traininputs = numpy.hstack((numpy.random.randn(2,100)-1.0,numpy.random.randn(2,100)+1.0))
    trainlabels = onehot(numpy.hstack((numpy.ones((100))*0,numpy.ones((100))*1)).astype("int")).T
    testinputs = numpy.hstack((numpy.random.randn(2,100)-1.0,numpy.random.randn(2,100)+1.0))
    testlabels = onehot(numpy.hstack((numpy.ones((100))*0,numpy.ones((100))*1)).astype("int")).T
    testinputs = numpy.hstack((numpy.random.randn(2,100)-1.0,numpy.random.randn(2,100)+1.0))
    print traininputs.shape
    '''
    
    #build and train a model:
    numsteps = 150
    model = logreg.Logreg(numclasses, d)
    model.train(x_train, y_train, 0.1, numsteps)
    
    # try model on train data:
    predictedlabels = model.classify(x_train)
    print 'error rate on the train set : '
    print np.sum( logreg.unhot(y_train.T) != logreg.unhot(predictedlabels.T)) / float((4178 - valid_set_size))
    print '---------------' 
    predictedlabels = model.classify(x_valid) 
    print 'true labels: '
    print logreg.unhot(y_valid.T)
    print 'predicted labels: '
    print logreg.unhot(predictedlabels.T)

    print 'error rate of the valid set : '
    print np.sum( logreg.unhot(y_valid.T) != logreg.unhot(predictedlabels.T)) / float(y_valid.shape[1])
    


