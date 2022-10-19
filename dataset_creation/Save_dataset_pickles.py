import sys
sys.path.insert(0, '../libraries')

from utils import load_from_directory_to_pickle,safe_mkdir

base_dir='./probav_data/'
#base_dir='/home/bordone/Superresolution/data/probav_data'
out_dir='./pickles/'
safe_mkdir(out_dir)
for band in ['NIR','RED']:
    load_from_directory_to_pickle(base_dir,out_dir,band=band)