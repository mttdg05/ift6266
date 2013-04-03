
# a modified version of __init.py__ made by Ian Goodfellow
import csv
import cPickle as pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('which_set', help = 'which set (train || test)')
parser.add_argument('path', help = 'set path')

args = parser.parse_args()

which_set = args.which_set
path = args.path

csv_file = open(path, 'r')
reader = csv.reader(csv_file)
# Discard header
row = reader.next()

y_list = []
X_list = []

for row in reader:
  if which_set == 'train':
    y_str, X_row_str = row
    y = int(y_str)
    y_list.append(y)
  else:
    X_row_str ,= row
  X_row_strs = X_row_str.split(' ')
  X_row = map(lambda x: float(x), X_row_strs)
  X_list.append(X_row)

X = np.asarray(X_list)


if which_set == 'train':
  y = np.asarray(y_list)
  '''
  one_hot = np.zeros((y.shape[0],7),dtype='float32')
  for i in xrange(y.shape[0]):
    one_hot[i,y[i]] = 1.
  y = one_hot
  '''
  # pickle the data
  pickle.dump(X, open('train_inputs.p', 'wb'))
  pickle.dump(y, open('train_labels.p', 'wb'))

else:
  y = None
  pickle.dump(X, open('test_inputs.p', 'wb'))





      
