#!/bin/sh

: '
# step 1 -> take patches with a stride = 1, whiten the data & save it in data/whiten_patches"patch_number".py
# To avoid ram problems -> create multiple python file & run them one by one
python make_patches_files.py test
python patches/test/make_test_batch0.py && python patches/test/make_test_batch1.py

# step 2 -> compute the encoder & pool. 
# create multiples files each containing a portion of the dataset and run them one by one.
python make_encod_pool_files.py test


python encod_pool/test/encod_pool0.py && python encod_pool/test/encod_pool1.py && python encod_pool/test/encod_pool2.py && python encod_pool/test/encod_pool3.py && python encod_pool/test/encod_pool4.py && python encod_pool/test/encod_pool5.py && python encod_pool/test/encod_pool6.py && python encod_pool/test/encod_pool7.py 



# After the pooling the dimensions is reduced to 4000. We have 4178  points. 
# A dataset of 4178 * 4000 is quite reasonable in terms of mememory -> aggregate it in one final dataset features.npy 
# Check -> maybe the 4000 features vectors will increase the chances of overfitting

python make_features_dataset.py test
'


# Now we that we have a dataset of "useful" features (who knows ...) -> 
# Train a simple one against all SVM (with sklearn ) & use on the test set. 
# Since the dataset is quite small ->  cross validation is used.

python classify.py

