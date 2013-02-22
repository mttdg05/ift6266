from pylearn2.utils import serial
import contest_dataset
from pylearn2.datasets import preprocessing

if __name__ == "__main__":
    train =  contest_dataset.ContestDataset (which_set="train", base_path='/home/a1/ContestDataset', start = 0, stop =  3678 )
    valid =  contest_dataset.ContestDataset (which_set="train", base_path='/home/a1/ContestDataset', start = 3678, stop = 4178 )
    pipeline = preprocessing.Pipeline()
    pipeline.items.append(preprocessing.GlobalContrastNormalization())
    pipeline.items.append(preprocessing.ZCA())
    train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    valid.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    train.use_design_loc('train_design.npy')
    valid.use_design_loc('valid_design.npy')
    serial.save('train_prep.pkl', train)
    serial.save('valid_prep.pkl', valid)
    
