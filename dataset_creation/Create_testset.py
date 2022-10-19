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
out_dataset='./testset'
safe_mkdir(out_dataset)
#base_dir='./probav_data'

band='NIR'
input_images_LR_test = np.load(os.path.join(dir_pickles_probav, 'LR_test_{0}.npy'.format(band)), allow_pickle=True)
mask_LR_test = np.load(os.path.join(dir_pickles_probav, 'LR_mask_{0}_test.npy'.format(band)), allow_pickle=True)

import matplotlib.pyplot as plt
plt.figure(figsize=[5,5])
plt.imshow((input_images_LR_test[0][0]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis


#transform in a list of numpy
input_images_LR_test=np.array([np.array(x) for x in input_images_LR_test])
mask_LR_test=np.array([np.array(x) for x in mask_LR_test])

#Find the indexes to remove with very high pixels
images_to_remove=[[i,j] for i,x in enumerate(input_images_LR_test) for j,z in enumerate(x) if (z>60000).any() ]
#generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
# to remove of that specific imageset
from collections import defaultdict
d=defaultdict(list)
for i in images_to_remove:
    d[i[0]].append(i[1])
    

for i in d.keys():
    input_images_LR_test[i]\
    =np.delete(input_images_LR_test[i],d[i],axis=0)
    
    mask_LR_test[i]\
    =np.delete(mask_LR_test[i],d[i],axis=0)

indexes=[i for i,x in enumerate(input_images_LR_test) if np.array(x).shape[0]<9]

#LR
input_images_LR_test=np.delete(input_images_LR_test,indexes,axis=0)
mask_LR_test=np.delete(mask_LR_test,indexes,axis=0)

input_images_LR_test_upsample=upsampling_without_aggregation_all_imageset(input_images_LR_test,scale=3)

mask_LR_test_upsample=upsampling_mask_all_imageset(mask_LR_test,scale=3)

input_images_LR_test_upsample_registered,\
mask_LR_test_upsample_registered,\
shifts_test,\
new_index_orders_test=registration_imageset_against_best_image_without_union_mask(input_images_LR_test_upsample,
                                                mask_LR_test_upsample,
                                                1)



#transform in a list of numpy
input_images_LR_test_upsample=np.array([np.array(x) for x in input_images_LR_test_upsample])
mask_LR_test_upsample=np.array([np.array(x) for x in mask_LR_test_upsample])

#Reorder the upsampled and not registered testset the way the upsdampled and registered testset has been ordered
#so that it matched the ordering of the shifts we computed during registration

input_images_LR_test_upsample=[imageset[new_index_orders_test[i]] for i,imageset in enumerate(input_images_LR_test_upsample)]
mask_LR_test_upsample=[imageset[new_index_orders_test[i]] for i,imageset in enumerate(mask_LR_test_upsample)]

#Find the indexes to remove considering we want to keep up to 4 pixel shift.
images_to_remove=[[i,j,z] for i,x in enumerate(shifts_test) for j,z in enumerate(x) if (np.abs(z)>4).any() ]
#generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
# to remove of that specific imageset
from collections import defaultdict

d=defaultdict(list)

for i in images_to_remove:
    d[i[0]].append(i[1])
    
for i in d.keys():
    input_images_LR_test_upsample[i]\
    =np.delete(input_images_LR_test_upsample[i],d[i],axis=0)
    
    mask_LR_test_upsample[i]\
    =np.delete(mask_LR_test_upsample[i],d[i],axis=0)

# Update also shifts array

for i in d.keys():
    shifts_test[i]\
    =np.delete(shifts_test[i],d[i],axis=0)

#shifts_test=np.delete(shifts_test,indexes,axis=0) NOT NEEDED


np.save(os.path.join(out_dataset,'dataset_{0}_LR_test.npy'.format(band)),input_images_LR_test_upsample,allow_pickle=True)
np.save(os.path.join(out_dataset,'dataset_{0}_mask_LR_test.npy'.format(band)),mask_LR_test_upsample,allow_pickle=True)
np.save(os.path.join(out_dataset,'shifts_test_{0}.npy'.format(band)),shifts_test,allow_pickle=True)


plt.figure(figsize=[8,8])
plt.imshow((input_images_LR_test[17][3]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.figure(figsize=[8,8])
plt.imshow((input_images_LR_test_upsample[17][3]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.figure(figsize=[8,8])
plt.imshow((input_images_LR_test_upsample[17][0]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.figure(figsize=[8,8])
plt.imshow((input_images_LR_test_upsample_registered[17][3]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

band='RED'
input_images_LR_test = np.load(os.path.join(dir_pickles_probav, 'LR_test_{0}.npy'.format(band)), allow_pickle=True)
mask_LR_test = np.load(os.path.join(dir_pickles_probav, 'LR_mask_{0}_test.npy'.format(band)), allow_pickle=True)

plt.figure(figsize=[5,5])
plt.imshow((input_images_LR_test[0][0]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

#transform in a list of numpy
input_images_LR_test=np.array([np.array(x) for x in input_images_LR_test])
mask_LR_test=np.array([np.array(x) for x in mask_LR_test])

#Find the indexes to remove with very high pixels
images_to_remove=[[i,j] for i,x in enumerate(input_images_LR_test) for j,z in enumerate(x) if (z>60000).any() ]
#generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
# to remove of that specific imageset
from collections import defaultdict
d=defaultdict(list)
for i in images_to_remove:
    d[i[0]].append(i[1])
    
for i in d.keys():
    input_images_LR_test[i]\
    =np.delete(input_images_LR_test[i],d[i],axis=0)
    
    mask_LR_test[i]\
    =np.delete(mask_LR_test[i],d[i],axis=0)

indexes=[i for i,x in enumerate(input_images_LR_test) if np.array(x).shape[0]<9]

#LR
input_images_LR_test=np.delete(input_images_LR_test,indexes,axis=0)
mask_LR_test=np.delete(mask_LR_test,indexes,axis=0)

input_images_LR_test_upsample=upsampling_without_aggregation_all_imageset(input_images_LR_test,scale=3)

mask_LR_test_upsample=upsampling_mask_all_imageset(mask_LR_test,scale=3)

input_images_LR_test_upsample_registered,\
mask_LR_test_upsample_registered,\
shifts_test,\
new_index_orders_test=registration_imageset_against_best_image_without_union_mask(input_images_LR_test_upsample,
                                                mask_LR_test_upsample,
                                                1)


#transform in a list of numpy
input_images_LR_test_upsample=np.array([np.array(x) for x in input_images_LR_test_upsample])
mask_LR_test_upsample=np.array([np.array(x) for x in mask_LR_test_upsample])

#Reorder the upsampled and not registered testset the way the upsdampled and registered testset has been ordered
#so that it matched the ordering of the shifts we computed during registration

input_images_LR_test_upsample=[imageset[new_index_orders_test[i]] for i,imageset in enumerate(input_images_LR_test_upsample)]
mask_LR_test_upsample=[imageset[new_index_orders_test[i]] for i,imageset in enumerate(mask_LR_test_upsample)]

#Find the indexes to remove considering we want to keep up to 4 pixel shift.
images_to_remove=[[i,j,z] for i,x in enumerate(shifts_test) for j,z in enumerate(x) if (np.abs(z)>4).any() ]
#generate dictionary with as key the index of the imageset and as value a list of indexes correpsonding to images
# to remove of that specific imageset
from collections import defaultdict

d=defaultdict(list)

for i in images_to_remove:
    d[i[0]].append(i[1])
    
for i in d.keys():
    input_images_LR_test_upsample[i]\
    =np.delete(input_images_LR_test_upsample[i],d[i],axis=0)
    
    mask_LR_test_upsample[i]\
    =np.delete(mask_LR_test_upsample[i],d[i],axis=0)

# Update also shifts array

for i in d.keys():
    shifts_test[i]\
    =np.delete(shifts_test[i],d[i],axis=0)

#shifts_test=np.delete(shifts_test,indexes,axis=0) NOT NEEDED

np.save(os.path.join(out_dataset,'dataset_{0}_LR_test.npy'.format(band)),input_images_LR_test_upsample,allow_pickle=True)
np.save(os.path.join(out_dataset,'dataset_{0}_mask_LR_test.npy'.format(band)),mask_LR_test_upsample,allow_pickle=True)
np.save(os.path.join(out_dataset,'shifts_test_{0}.npy'.format(band)),shifts_test,allow_pickle=True)

plt.figure(figsize=[8,8])
plt.imshow((input_images_LR_test[17][3]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.figure(figsize=[8,8])
plt.imshow((input_images_LR_test_upsample[17][3]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.figure(figsize=[8,8])
plt.imshow((input_images_LR_test_upsample[17][0]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.figure(figsize=[8,8])
plt.imshow((input_images_LR_test_upsample_registered[17][3]).squeeze(), cmap = 'gray', interpolation = 'none')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis