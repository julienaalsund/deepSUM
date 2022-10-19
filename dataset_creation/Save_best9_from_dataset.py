import sys
sys.path.insert(0, '../libraries')


import random
import numpy as np
from collections import defaultdict
import progressbar
from utils import safe_mkdir
from dataloader import load_dataset

directory='./dataset_light_best9/'
safe_mkdir(directory)
dataset_dir='./dataset/'

n_chunks=5
for band in ['NIR','RED']:

    gen=load_dataset(dataset_dir,n_chunks,band,num_images=9,how='best')

    for i in range(n_chunks):
        dataset_dict=next(gen)
        
        batch_training=dataset_dict['training']
        batch_training_mask=dataset_dict['training_mask']
        batch_training_y=dataset_dict['training_y']
        batch_mask_train_y=dataset_dict['training_mask_y']
        shifts=dataset_dict['shifts']
        
        batch_validation=dataset_dict['validation']
        batch_validation_mask=dataset_dict['validation_mask']
        batch_validation_y=dataset_dict['validation_y']
        batch_mask_valid_y=dataset_dict['validation_mask_y']
        
        shifts_valid=dataset_dict['shifts_valid']
        norm_validation=dataset_dict['norm_validation']
        
        batch_training_mask=batch_training_mask.astype('bool')
        batch_mask_train_y=batch_mask_train_y.astype('bool')
        batch_validation_mask=batch_validation_mask.astype('bool')
        batch_mask_valid_y=batch_mask_valid_y.astype('bool')
        
        
        np.save(directory+'{0}_dataset_{1}_patch_LR_best9.npy'.format(i,band),batch_training,allow_pickle=True)
        np.save(directory+'{0}_dataset_{1}_patch_HR_best9.npy'.format(i,band),batch_training_y,allow_pickle=True)
        np.save(directory+'{0}_dataset_{1}_patch_mask_LR_best9.npy'.format(i,band),batch_training_mask,allow_pickle=True)
        np.save(directory+'{0}_dataset_{1}_patch_mask_HR_best9.npy'.format(i,band),batch_mask_train_y,allow_pickle=True)
        np.save(directory+'{0}_shifts_patch_{1}_best9.npy'.format(i,band),shifts,allow_pickle=True)
        
        print('Saving {0}'.format(i))
        if i==0:
            np.save(directory+'dataset_{0}_LR_valid_best9.npy'.format(band),batch_validation,allow_pickle=True)
            np.save(directory+'dataset_{0}_HR_valid_best9.npy'.format(band),batch_validation_y,allow_pickle=True)
            np.save(directory+'dataset_{0}_mask_LR_valid_best9.npy'.format(band),batch_validation_mask,allow_pickle=True)
            np.save(directory+'dataset_{0}_mask_HR_valid_best9.npy'.format(band),batch_mask_valid_y,allow_pickle=True)
            
            #normalization
            np.save(directory+'norm_'+band+'.npy',norm_validation,allow_pickle=True)
            
            np.save(directory+'shifts_valid_{0}_best9.npy'.format(band),shifts_valid,allow_pickle=True)
            