from pylearn2.utils import serial
import contest_dataset
from pylearn2.datasets import preprocessing

if __name__ == "__main__":
    train =  contest_dataset.ContestDataset (which_set="train", base_path='/home/a1/ContestDataset', start = 0, stop =  3500 )
    valid =  contest_dataset.ContestDataset (which_set="train", base_path='/home/a1/ContestDataset', start = 3500, stop = 4178 )
    pipeline = preprocessing.Pipeline()
    #pipeline.items.append(preprocessing.GlobalContrastNormalization())
    pipeline.items.append(preprocessing.Standardize())
    #pipeline.items.append(preprocessing.ZCA())
    train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    valid.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    train.use_design_loc('ftrain_design.npy')
    valid.use_design_loc('fvalid_design.npy')
    serial.save('ftrain_prep.pkl', train)
    serial.save('fvalid_prep.pkl', valid)
    
