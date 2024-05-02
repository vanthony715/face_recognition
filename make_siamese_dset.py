# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.utils import create_pairs
from utils.utils import make_dir

import gc
gc.collect()

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    ##define paths
    datapath = 'C:/Users/vanth/DnHFaces/open_data_set/'
    writepath = datapath + 'siamese_dset/'
    write_train = writepath + 'train/'
    write_test = writepath + 'test/'

    num_per_pair_test = 8
    num_pairs_test = 8
    num_per_pair_train = 8
    num_pairs_train = 80

    '''
    Make dataset csv: ID, Imgname
    '''
    file_dict = {'id': [], 'label': []}
    for i, file in enumerate(os.listdir(datapath + 'photos_all_faces/')):
        file_dict['id'].append(file.replace('.jpg', ''))
        file_dict['label'].append(file.split('_')[0])
    ##write data to disk for reference
    df = pd.DataFrame(file_dict)
    df.to_csv(datapath + 'trainLabelsSiam.csv', index=False)

    ##make new top-level directories
    make_dir(write_train)
    make_dir(write_test)

    ldf = pd.read_csv(datapath + 'trainLabelsSiam.csv') #open csv as a test
    files = os.listdir(datapath + 'photos_all_faces/')
    files = [datapath + 'photos_all_faces/' + i for i in files]
    print('Total Num Files: ', len(files))

    ##get filepaths and classes
    file_dict = {'filepath': [], 'class': []}
    for i, filepath in enumerate(files):
        idx = filepath.split('/')[-1].split('.jpg')[0]
        clss = ldf.loc[ldf.id == idx].label.values[0]
        file_dict['filepath'].append(filepath)
        file_dict['class'].append(clss)

    file_df = pd.DataFrame(file_dict)

    ##split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(file_df, file_df,
                                                        test_size=0.20,
                                                        random_state=42)
    print('Train dset size: ', len(X_train))
    print('Test dset size: ', len(X_test))

    ##create pairs genuine/imposter pairs with mode indicating which
    test_df = create_pairs(X_test, num_per_pair_test, num_pairs_test, write_test)
    train_df = create_pairs(X_train, num_per_pair_train, num_pairs_train,
                            write_train)

    ##write to csv for reference of data/dataloader objects
    test_df.to_csv(writepath + 'testdata.csv', header=False, index=False)
    train_df.to_csv(writepath + 'traindata.csv', header=False, index=False)
