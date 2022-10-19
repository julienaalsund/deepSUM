import sys
sys.path.insert(0, '../libraries')


from dataloader import load_dataset,create_patch_dataset_return_shifts
from utils import safe_mkdir,upsampling_without_aggregation_all_imageset,upsampling_mask_all_imageset,registration_imageset_against_best_image_without_union_mask
import numpy as np
import pandas as pd
from skimage.feature import register_translation
import os
import glob
from collections import defaultdict

dir_pickles_probav='./pickles/'
out_dataset='./dataset'
safe_mkdir(out_dataset)
base_dir='./probav_data'
#base_dir='/home/bordone/Superresolution/data/probav_data/'

split=0.7
band='RED'
input_images_LR = np.load(os.path.join(dir_pickles_probav, 'LR_dataset_{0}.npy'.format(band)))
mask_LR = np.load(os.path.join(dir_pickles_probav, 'LR_mask_{0}.npy'.format(band)))
input_images_HR = np.load(os.path.join(dir_pickles_probav, 'HR_dataset_{0}.npy'.format(band)))
mask_HR = np.load(os.path.join(dir_pickles_probav, 'HR_mask_{0}.npy'.format(band)))

#To Compute the PSNR
#norm baseline for each imageset to normalize cPSNR
df_norm=pd.read_csv(os.path.join(base_dir, 'norm.csv'),sep=' ',header=None)
df_norm.columns=['set','norm']
train_dir = os.path.join(base_dir, 'train/{0}'.format(band))
dir_list=sorted([os.path.basename(x) for x in glob.glob(train_dir+'/imgset*')])
norm=df_norm.loc[df_norm['set'].isin(dir_list)]['norm'].values
norm=norm.reshape([norm.shape[0],1])

from sklearn.utils import shuffle

input_images_LR,mask_LR,input_images_HR,mask_HR,norm = shuffle(input_images_LR,
                                                          mask_LR,
                                                          input_images_HR,
                                                          mask_HR, 
                                                          norm,
                                                          random_state=1)

#Split training set and validation set
N_training_samples=int(split*len(input_images_LR))
input_images_LR_train,input_images_LR_valid=input_images_LR[0:N_training_samples],input_images_LR[N_training_samples:]
mask_LR_train,mask_LR_valid=mask_LR[0:N_training_samples],mask_LR[N_training_samples:]
input_images_HR_train,input_images_HR_valid=input_images_HR[0:N_training_samples],input_images_HR[N_training_samples:]
mask_HR_train,mask_HR_valid=mask_HR[0:N_training_samples],mask_HR[N_training_samples:]

norm_training,norm_validation=norm[0:N_training_samples],norm[N_training_samples:]

import matplotlib.pyplot as plt
plt.figure(figsize=[5,5])
plt.imshow((input_images_LR_train[0][0]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#transform in a list of numpy
input_images_LR_train=np.array([np.array(x) for x in input_images_LR_train])
mask_LR_train=np.array([np.array(x) for x in mask_LR_train])

#Find the indexes to remove with very high pixels
images_to_remove=[[i,j] for i,x in enumerate(input_images_LR_train) for j,z in enumerate(x) if (z>60000).any() ]
#generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
# to remove of that specific imageset
d=defaultdict(list)
for i in images_to_remove:
    d[i[0]].append(i[1])

for i in d.keys():
    input_images_LR_train[i]\
    =np.delete(input_images_LR_train[i],d[i],axis=0)
    
    mask_LR_train[i]\
    =np.delete(mask_LR_train[i],d[i],axis=0)

[x.shape for x in input_images_LR_train if x.shape[0]<9]

indexes=[i for i,x in enumerate(input_images_LR_train) if np.array(x).shape[0]<9]

#LR
input_images_LR_train=np.delete(input_images_LR_train,indexes,axis=0)
mask_LR_train=np.delete(mask_LR_train,indexes,axis=0)

#HR
input_images_HR_train=np.delete(input_images_HR_train,indexes,axis=0)
mask_HR_train=np.delete(mask_HR_train,indexes,axis=0)

input_images_LR_train_upsample=upsampling_without_aggregation_all_imageset(input_images_LR_train,scale=3)

mask_LR_train_upsample=upsampling_mask_all_imageset(mask_LR_train,scale=3)

input_images_LR_train_upsample_registered,\
mask_LR_train_upsample_registered,\
shifts,\
new_index_orders=registration_imageset_against_best_image_without_union_mask(input_images_LR_train_upsample,
                                                mask_LR_train_upsample,
                                                1)

new_index_orders[17]

shifts[17]

#transform in a list of numpy
input_images_LR_train_upsample=np.array([np.array(x) for x in input_images_LR_train_upsample])
mask_LR_train_upsample=np.array([np.array(x) for x in mask_LR_train_upsample])

#Reorder the training set not upsampled and not registered the way the training upsdampled and registered has been ordered
#so that it matched the ordering of the shifts we computed during registration

input_images_LR_train_upsample=[imageset[new_index_orders[i]] for i,imageset in enumerate(input_images_LR_train_upsample)]
mask_LR_train_upsample=[imageset[new_index_orders[i]] for i,imageset in enumerate(mask_LR_train_upsample)]

#Find the indexes to remove considering we want to keep up to 4 pixel shift. 
images_to_remove=[[i,j,z] for i,x in enumerate(shifts) for j,z in enumerate(x) if (np.abs(z)>4).any() ]
#generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
# to remove of that specific imageset
from collections import defaultdict

d=defaultdict(list)

for i in images_to_remove:
    d[i[0]].append(i[1])
    
for i in d.keys():
    input_images_LR_train_upsample[i]\
    =np.delete(input_images_LR_train_upsample[i],d[i],axis=0)
    
    mask_LR_train_upsample[i]\
    =np.delete(mask_LR_train_upsample[i],d[i],axis=0)

[x.shape for x in input_images_LR_train_upsample if x.shape[0]<9]

indexes=[i for i,x in enumerate(input_images_LR_train_upsample) if np.array(x).shape[0]<9]

#LR
input_images_LR_train_upsample=np.delete(input_images_LR_train_upsample,indexes,axis=0)
mask_LR_train_upsample=np.delete(mask_LR_train_upsample,indexes,axis=0)

#HR
input_images_HR_train=np.delete(input_images_HR_train,indexes,axis=0)
mask_HR_train=np.delete(mask_HR_train,indexes,axis=0)

# Update also shifts array
for i in d.keys():
    shifts[i]\
    =np.delete(shifts[i],d[i],axis=0)

shifts=np.delete(shifts,indexes,axis=0)

n_chuncks=5
for i in range(n_chuncks):
    dataset_patch=create_patch_dataset_return_shifts(input_images_LR_train_upsample,
                                                     input_images_HR_train,
                                                     mask_LR_train_upsample,
                                                     mask_HR_train,
                                                     shifts,
                                                     patch_size=96,
                                                     num_patches_per_set=20,
                                                     scale=1,
                                                     smart_patching=True
                                                     )
    
    input_images_LR_patch=dataset_patch['training_patch']
    input_images_HR_patch=dataset_patch['training_y_patch']
    mask_LR_patch=dataset_patch['training_mask_patch']
    mask_HR_patch=dataset_patch['training_mask_y_patch']
    shifts_patch=dataset_patch['shifts']
    coordinates=dataset_patch['coordinates']
    
    np.save(os.path.join(out_dataset,'{0}_dataset_{1}_patch_LR.npy'.format(i,band)),input_images_LR_patch,allow_pickle=True)
    np.save(os.path.join(out_dataset,'{0}_dataset_{1}_patch_HR.npy'.format(i,band)),input_images_HR_patch,allow_pickle=True)
    np.save(os.path.join(out_dataset,'{0}_dataset_{1}_patch_mask_LR.npy'.format(i,band)),mask_LR_patch,allow_pickle=True)
    np.save(os.path.join(out_dataset,'{0}_dataset_{1}_patch_mask_HR.npy'.format(i,band)),mask_HR_patch,allow_pickle=True)
    np.save(os.path.join(out_dataset,'{0}_shifts_patch_{1}.npy'.format(i,band)),shifts_patch,allow_pickle=True)
    np.save(os.path.join(out_dataset,'{0}_coordinates_{1}.npy'.format(i,band)),coordinates,allow_pickle=True)

import matplotlib.pyplot as plt
plt.figure(figsize=[5,5])
plt.imshow((input_images_LR_patch[14][8]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.figure(figsize=[5,5])
plt.imshow((input_images_LR_patch[14][13]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.figure(figsize=[5,5])
plt.imshow((mask_LR_patch[14][13]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.figure(figsize=[5,5])
plt.imshow((mask_LR_patch[23][0]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.figure(figsize=[5,5])
plt.imshow((input_images_HR_patch[14]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#transform in a list of numpy
input_images_LR_valid=np.array([np.array(x) for x in input_images_LR_valid])
mask_LR_valid=np.array([np.array(x) for x in mask_LR_valid])

#Find the indexes to remove with very high pixels
images_to_remove=[[i,j] for i,x in enumerate(input_images_LR_valid) for j,z in enumerate(x) if (z>60000).any() ]
#generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
# to remove of that specific imageset
from collections import defaultdict

d=defaultdict(list)

for i in images_to_remove:
    d[i[0]].append(i[1])
    
for i in d.keys():
    input_images_LR_valid[i]\
    =np.delete(input_images_LR_valid[i],d[i],axis=0)
    
    mask_LR_valid[i]\
    =np.delete(mask_LR_valid[i],d[i],axis=0)

[x.shape for x in input_images_LR_valid if x.shape[0]<9]

indexes=[i for i,x in enumerate(input_images_LR_valid) if np.array(x).shape[0]<9]

#LR
input_images_LR_valid=np.delete(input_images_LR_valid,indexes,axis=0)
mask_LR_valid=np.delete(mask_LR_valid,indexes,axis=0)

#HR
input_images_HR_valid=np.delete(input_images_HR_valid,indexes,axis=0)
mask_HR_valid=np.delete(mask_HR_valid,indexes,axis=0)

#update the baseline normalization
norm_validation=np.delete(norm_validation,indexes,axis=0)

input_images_LR_valid_upsample=upsampling_without_aggregation_all_imageset(input_images_LR_valid,scale=3)

mask_LR_valid_upsample=upsampling_mask_all_imageset(mask_LR_valid,scale=3)

input_images_LR_valid_upsample_registered,\
mask_LR_valid_upsample_registered,\
shifts_valid,\
new_index_orders_valid=registration_imageset_against_best_image_without_union_mask(input_images_LR_valid_upsample,
                                                mask_LR_valid_upsample,
                                                1)


#transform in a list of numpy
input_images_LR_valid_upsample=np.array([np.array(x) for x in input_images_LR_valid_upsample])
mask_LR_valid_upsample=np.array([np.array(x) for x in mask_LR_valid_upsample])

#Reorder the training set not upsampled and not registered the way the training upsdampled and registered has been ordered
#so that it matched the ordering of the shifts we computed during registration

input_images_LR_valid_upsample=[imageset[new_index_orders_valid[i]] for i,imageset in enumerate(input_images_LR_valid_upsample)]
mask_LR_valid_upsample=[imageset[new_index_orders_valid[i]] for i,imageset in enumerate(mask_LR_valid_upsample)]

# Find the indexes to remove

images_to_remove=[[i,j,z] for i,x in enumerate(shifts_valid) for j,z in enumerate(x) if (np.abs(z)>4).any() ]
#generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
# to remove of that specific imageset
from collections import defaultdict

d=defaultdict(list)

for i in images_to_remove:
    d[i[0]].append(i[1])
    
for i in d.keys():
    input_images_LR_valid_upsample[i]\
    =np.delete(input_images_LR_valid_upsample[i],d[i],axis=0)
    
    mask_LR_valid_upsample[i]\
    =np.delete(mask_LR_valid_upsample[i],d[i],axis=0)

[x.shape for x in input_images_LR_valid_upsample if x.shape[0]<9]

indexes=[i for i,x in enumerate(input_images_LR_valid_upsample) if np.array(x).shape[0]<9]

#LR
input_images_LR_valid_upsample=np.delete(input_images_LR_valid_upsample,indexes,axis=0)
mask_LR_valid_upsample=np.delete(mask_LR_valid_upsample,indexes,axis=0)

#HR
input_images_HR_valid=np.delete(input_images_HR_valid,indexes,axis=0)
mask_HR_valid=np.delete(mask_HR_valid,indexes,axis=0)

# Update also shifts array

for i in d.keys():
    shifts_valid[i]\
    =np.delete(shifts_valid[i],d[i],axis=0)

shifts_valid=np.delete(shifts_valid,indexes,axis=0)

#update the baseline normalization
norm_validation=np.delete(norm_validation,indexes,axis=0)

np.save(os.path.join(out_dataset,'dataset_{0}_LR_valid.npy'.format(band)),input_images_LR_valid_upsample,allow_pickle=True)
np.save(os.path.join(out_dataset,'dataset_{0}_HR_valid.npy'.format(band)),input_images_HR_valid,allow_pickle=True)
np.save(os.path.join(out_dataset,'dataset_{0}_mask_LR_valid.npy'.format(band)),mask_LR_valid_upsample,allow_pickle=True)
np.save(os.path.join(out_dataset,'dataset_{0}_mask_HR_valid.npy'.format(band)),mask_HR_valid,allow_pickle=True)
np.save(os.path.join(out_dataset,'shifts_valid_{0}.npy'.format(band)),shifts_valid,allow_pickle=True)

norm_validation.dump(os.path.join(out_dataset,'norm_'+band+'.npy'))