import numpy as np
import os
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("which_set", help = "which_set : train || test")
args = parser.parse_args()
which_set = str(args.which_set)

if which_set != 'train' and which_set != 'test': raise ValueError("The set must be train or test!")

data = None
filenames = []

if which_set == 'train' :
    for filename in sorted(os.listdir("data/pooled_features/train")):
    	if filename.startswith("train_pooled_features") and filename.endswith(".npy") :
            filenames.append(filename)

    features = None
    for filename in filenames :
        filename = ''.join(["data/pooled_features/train/", filename])
        if features is None :
            features = np.load(filename)
        else : 
            features = np.vstack((features, np.load(filename)))
    
    np.save('data/features/train_features', features)

else :
    for filename in sorted(os.listdir("data/pooled_features/test")):
    	if filename.startswith("test_pooled_features") and filename.endswith(".npy") :
            filenames.append(filename)

    features = None
    for filename in filenames :
        filename = ''.join(["data/pooled_features/test/", filename])
        if features is None :
            features = np.load(filename)
        else : 
            features = np.vstack((features, np.load(filename)))

    np.save('data/features/test_features', features)
