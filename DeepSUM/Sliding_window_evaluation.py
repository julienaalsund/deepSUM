#!/usr/bin/env python
# coding: utf-8

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from collections import defaultdict
import numpy as np
from DeepSUM_network import SR_network
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import json

config_file='./config_files/DeepSUM_config_NIR.json'
with open(config_file) as json_data_file:
    data = json.load(json_data_file)

        
tf.compat.v1.reset_default_graph()
config=defaultdict()
config['lr']= data['hyperparameters']['lr']
config['batch_size'] =  data['hyperparameters']['batch_size']
config['base_dir'] = data['others']['base_dir']
config['skip_step'] = data['others']['skip_step']
config['channels'] = data['others']['channels']
config['T_in'] = data['others']['T_in'] 
config['R'] = data['others']['R']
config['full'] = True
config['patch_size_HR'] = data['others']['patch_size_HR']
config['patch_size_LR'] = data['others']['patch_size_LR']
config['border'] = data['others']['border']
config['spectral_band']=data['others']['spectral_band']
config['RegNet_pretrain_dir']=data['others']['RegNet_pretrain_dir']
config['SISRNet_pretrain_dir']=data['others']['SISRNet_pretrain_dir']
config['dataset_path']=data['others']['dataset_path']
config['n_chunks']=data['others']['n_chunks']
config['mu']=data['others']['mu']
config['sigma']=data['others']['sigma']
config['sigma_rescaled']=data['others']['sigma_rescaled']

config['tensorboard_dir'] = 'DeepSUM_'+config['spectral_band']+'_lr_'+str(config['lr'])+'_bsize_'+str(config['batch_size'])

model = SR_network(config)

model.build()
    
step=model.train(n_epochs=0)

y=tf.placeholder('float32',shape=[None,1,384,384,1],name='y')
upsampled_x=tf.placeholder('float32',shape=[None,1,384,384,1],name='upsampled_x')
mask_y=tf.placeholder('float32',shape=[None,1,384,384,1],name='mask_y')
norm_baseline=tf.placeholder('float32',shape=[None,1],name='norm_baseline')

patch_size=384
border=3


y_masked_hat=(upsampled_x*model.sigma_rescaled)+model.mu#tf.multiply(upsampled_x,mask_y)
y_masked=(y*model.sigma_rescaled)+model.mu#tf.multiply(y,mask_y)
#Crop
s1=tf.shape(y_masked_hat)
s2=tf.shape(y_masked)
labels=tf.reshape(y_masked,shape=[s1[0],s1[2],s1[3],s1[4]])
predictions=tf.reshape(y_masked_hat,shape=[s2[0],s2[2],s2[3],s2[4]])

#cropped_predictions=tf.image.central_crop(image=predictions,central_fraction=0.984375)
cropped_predictions=predictions[:,border:patch_size-border,border:patch_size-border]

#cropped_labels=labels[:,0:0+378,0:0+378]
#All mse
X=[]
for i in range((2*border)+1):
    for j in range((2*border)+1):
        
        cropped_labels=labels[:,i:i+(patch_size-(2*border)),j:j+(patch_size-(2*border))]
        cropped_mask_y=mask_y[:,:,i:i+(patch_size-(2*border)),j:j+(patch_size-(2*border))]
        
        
        cropped_predictions_masked=cropped_predictions*tf.squeeze(cropped_mask_y,axis=1)
        cropped_labels_masked=cropped_labels*tf.squeeze(cropped_mask_y,axis=1)
        
        
        #bias brightness
        b=(1.0/tf.reduce_sum(cropped_mask_y,axis=[2,3,4]))*tf.reduce_sum(cropped_labels_masked-cropped_predictions_masked,axis=[1,2])
        b=tf.reshape(b,[s1[0],1,1,1])
        corrected_cropped_predictions=cropped_predictions_masked+b
        
        corrected_cropped_predictions=corrected_cropped_predictions*tf.squeeze(cropped_mask_y,axis=1)
        
        corrected_mse=(1.0/tf.reduce_sum(cropped_mask_y,axis=[2,3,4]))*tf.reduce_sum(tf.square(cropped_labels_masked-corrected_cropped_predictions),axis=[1,2])
        #cPSNR=-10*tf.log(corrected_mse)/tf.log(10.0)
        cPSNR=10*tf.log((65535**2)/corrected_mse)/tf.log(10.0)
        X.append(cPSNR) 


X=tf.stack(X)

max_cPSNR=tf.reduce_max(X,axis=0)

score=norm_baseline/max_cPSNR
score=tf.reduce_mean(score)

def sliding_window(band,directory,n_slides_list=[]):
    '''
    directory from where to load the validation dataset (validation with all images in an imageset!!!!)
    '''
    
    
    input_images_LR_valid=np.load(directory+'dataset_{0}_LR_valid.npy'.format(band),allow_pickle=True)
    input_images_HR_valid=np.load(directory+'dataset_{0}_HR_valid.npy'.format(band),allow_pickle=True)
    mask_LR_valid=np.load(directory+'dataset_{0}_mask_LR_valid.npy'.format(band),allow_pickle=True)
    mask_HR_valid=np.load(directory+'dataset_{0}_mask_HR_valid.npy'.format(band),allow_pickle=True)
    
    shifts_valid=np.load(directory+'shifts_valid_{0}.npy'.format(band),allow_pickle=True)
    #normalization
    norm_validation=np.load(directory+'norm_'+band+'.npy',allow_pickle=True)
    
    input_images_HR_valid=input_images_HR_valid.reshape([input_images_HR_valid.shape[0],1,384,384,1])
    mask_HR_valid=mask_HR_valid.reshape([mask_HR_valid.shape[0],1,384,384,1])
    input_images_HR_valid=(input_images_HR_valid-model.mu)/model.sigma_rescaled
    
    #import pandas as pd
    #df_score=pd.DataFrame
    
    SR_images_all_slides={}
    
    mean_scores={}
    for n_slides in n_slides_list:
        print(n_slides)
        #Order by mask
        indexes=[]
        
        for image_set in mask_LR_valid:
            indexes.append(np.argsort(np.sum(np.array(image_set[0:]),axis=(1,2)))[::-1])
            
        input_images_LR_valid=[image_set[indexes_set] for image_set,indexes_set in zip(input_images_LR_valid,indexes)]
        mask_LR_valid=np.array([image_set[indexes_set] for image_set,indexes_set in zip(mask_LR_valid,indexes)])
        #####
        
        
        
        #val_batch_size=1
        SR_images=np.zeros([len(input_images_LR_valid),1,384,384,1])
        for m in range(0,len(input_images_LR_valid)):
        
            imageset=np.array(input_images_LR_valid[m])
            imageset_mask=np.array(mask_LR_valid[m])
            imageset_shift=np.array(shifts_valid[m])
            
            #filter some images based on the mask
            print(imageset.shape)
            percentage=0.9
            while True:
                indexes_0=np.argwhere((np.sum(imageset_mask[0:],axis=(1,2))/(384*384))>percentage).squeeze(axis=1)
                indexes_0=indexes_0 if indexes_0.ndim>0 else np.array([]) 
                
                
                indexes=np.array(list(indexes_0))
                if indexes.size>=9:
                    imageset=imageset[indexes]
                    print(imageset.shape)
                    imageset_mask=imageset_mask[indexes]
                    imageset_shift=imageset_shift[indexes]
                    break
                else:
                    
                    percentage-=0.05
                    continue
        
        
            
            #imageset_HR=input_images_HR_valid[i]
            #imageset_mask_HR=mask_HR_valid[i]
            #Maybe HERE WE CAN REMOVE VERY BAD LR IMAGES
            len_imageset=np.shape(imageset)[0]
            
            temporal_dim=9
            upper_bound=n_slides
            if len_imageset-temporal_dim+1>upper_bound:
                size=upper_bound+1
            else:
                size=len_imageset-temporal_dim+1
                
            SR_imageset=np.zeros([size,1,384,384,1])
            for n in range(0,size):
                
                imageset_9=imageset[n:n+temporal_dim]
                #print(imageset_9.shape)
                #imageset_9=np.concatenate([np.expand_dims(reference_image,axis=0),imageset_9])
                imageset_9=np.expand_dims(imageset_9,axis=0)
                imageset_9=np.expand_dims(imageset_9,axis=-1)
                
                imageset_9_mask=imageset_mask[n:n+temporal_dim]
                #imageset_9_mask=np.concatenate([np.expand_dims(reference_mask,axis=0),imageset_9_mask])
                imageset_9_mask=np.expand_dims(imageset_9_mask,axis=0)
                imageset_9_mask=np.expand_dims(imageset_9_mask,axis=-1)
                
                ########################Register the mask #############
                imageset_9_mask=np.round(imageset_9_mask)
                imageset_9_mask=imageset_9_mask.astype('bool')
                
                for j in range(imageset_9_mask.shape[1]):
                    shifted_mask=imageset_9_mask[:,j]
                    corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), imageset_shift[j])
                    corrected_mask = np.fft.ifftn(corrected_mask)
                    corrected_mask = corrected_mask.reshape([1,np.shape(corrected_mask)[0],np.shape(corrected_mask)[1],1])
                    imageset_9_mask[:,j]=np.round(corrected_mask)
                ##############Compute coefficients for filling images where masked
                sh=imageset_9_mask.shape
                fill_coeff_valid=np.ones([sh[0],sh[1],sh[1],sh[2],sh[3],sh[4]],dtype='bool')
                for i in range(0,9):
                    fill_coeff_valid[:,:,i]=np.expand_dims(imageset_9_mask[:,i],axis=1)
        
                for i in range(0,9):
                    for j in range(i+1,9):
                        rows_indexes=[k for k in range(0,9) if k!=(j)]
                        #print(rows_indexes)
                        fill_coeff_valid[:,rows_indexes,j]=fill_coeff_valid[:,rows_indexes,j]*np.expand_dims(1-imageset_9_mask[:,i],axis=1)
                
                for i in range(1,9):
                    fill_coeff_valid[:,i,0:i]=fill_coeff_valid[:,i,0:i]*np.expand_dims(1-imageset_9_mask[:,i],axis=1)
                
                #We need to fill in the regions where all the masks are zero. In this case we decide to uncover the hidden regions of
                #the considered image by turning the mask to 1 in those regions.
                f=np.sum(fill_coeff_valid,axis=2)
                #[b,9,W,H,1]
                fill_coeff_valid[:,range(9),range(9),:,:,:]=fill_coeff_valid[:,range(9),range(9),:,:,:]+np.logical_not(f)[:,range(9),:,:,:]
                    
                ####################
                
                
                imageset_9=(imageset_9-model.mu)/model.sigma
                #imageset_HR=(imageset_HR-model.mu)/model.sigma_rescaled
        
            
                
                SR_image=model.sess.run(model.logits,feed_dict={
                                                              #model.y_filters:model.y_filters_valid_dyn[(i-1)*val_batch_size:i*val_batch_size],
                                                              #model.y:imageset_HR,
                                                              model.x:imageset_9,
                                                              model.fill_coeff:fill_coeff_valid,
                                                              #model.mask_y:imageset_9_mask
                                                            })
                
                
                
                
                SR_imageset[n]=SR_image
                
            #Register all images in SR_imageset with respect to the first
            #Compute the shift
            #SR_imageset_registered=np.zeros_like(SR_imageset)
            SR_imageset_registered=np.empty([0,1,384,384,1])
            for z in range(SR_imageset.shape[0]):
                #we consider the first image in the set as the reference image
                reference_image=SR_imageset[0]
                shifted_image=SR_imageset[z]
                
                shift, error, diffphase = register_translation(reference_image.squeeze(), shifted_image.squeeze(),upsample_factor=1)
                if (np.abs(shift)>4).any():
                    print('stop')
                    print(shift)
                    continue
                
                ###Image
                #shift is applied to the original image from the batch_training variable, in the fourier domain
                corrected_image = fourier_shift(np.fft.fftn(shifted_image.squeeze()), shift)
                corrected_image = np.fft.ifftn(corrected_image)
                corrected_image = corrected_image.reshape([1,1,384,384,1])
                #SR_imageset_registered[z]=corrected_image
                SR_imageset_registered=np.append(SR_imageset_registered,corrected_image,axis=0)   
            
            
            SR_image=np.mean(SR_imageset_registered,axis=0,keepdims=True)
            SR_images[m]=SR_image
            print('Image number {0}'.format(m))
            
            SR_images_all_slides[n_slides]=SR_images
        

            
        score_list=[]
        val_batch_size=1
        for i in range(1,int(SR_images.shape[0]/val_batch_size)+1):
        
            out=model.sess.run(max_cPSNR,feed_dict={
                                                y:input_images_HR_valid[(i-1)*val_batch_size:i*val_batch_size],
                                                upsampled_x:SR_images[(i-1)*val_batch_size:i*val_batch_size],
                                                norm_baseline:norm_validation[(i-1)*val_batch_size:i*val_batch_size],
                                                mask_y:mask_HR_valid[(i-1)*val_batch_size:i*val_batch_size]
                                                })
            
            

            score_list.append(out)
            
        mean_scores[n_slides]=score_list
    
    
    return mean_scores,SR_images_all_slides

n_slides=1
scores,SR_images_all_slides=sliding_window(config['spectral_band'],'../dataset_creation/dataset/',n_slides_list=range(n_slides+1))

print(np.mean(scores[1]))