# -*- coding: utf-8 -*-
"""
@author avasque1@jh.edu
"""
import os
import numpy as np
import pandas as pd

import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

class SiameseCustomImageDataFromFolder(Dataset):
    '''
    Custom Dataset for Siamese Data
    '''
    def __init__(self, train_csv, train_dir, transform=None):
        self.train_df = pd.read_csv(train_csv)
        self.train_df.columns = ['image1', 'image2', 'image3']
        self.train_dir =  train_dir
        self.transform = transform

    def __getitem__(self,index):
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])
        img1 = Image.fromarray(io.imread(image1_path))
        img2 = Image.fromarray(io.imread(image2_path))

        #apply image transormations
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([int(self.train_df.iat[index, 2])],dtype=np.float32))

    def __len__(self):
        return len(self.train_df)
