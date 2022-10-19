import tensorflow as tf
from collections import defaultdict

from DeepSUM_network import SR_network

import json

# Change this to train the RED model
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
config['full'] = data['others']['full']
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
    
    
step=model.train(n_epochs=10)