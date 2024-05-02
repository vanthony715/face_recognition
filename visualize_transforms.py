# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

##define libraries
import gc, time
gc.collect()

import sys
sys.path.append('srcutils/')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
plt.ion()

import torch
# torch.manual_seed(17)

import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision import datasets, tv_tensors

from siamese_custom_image_data_from_folder import SiameseCustomImageDataFromFolder


##define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    gc.collect()

    ## define paths and load dataset
    train_dir = "C:/Users/vanth/DnHFaces/open_data_set/siamese_dset/train"
    test_dir = "C:/Users/vanth/DnHFaces/open_data_set/siamese_dset/test"
    train_csv = "C:/Users/vanth/DnHFaces/open_data_set/siamese_dset/traindata.csv"
    test_csv =  "C:/Users/vanth/DnHFaces/open_data_set/siamese_dset/testdata.csv"

    ##define device
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    ##define image transforms to avoid overfitting data
    train_transforms = T.Compose([
                        T.RandomResizedCrop(64),  # Randomly resize and crop to 64x64
                        T.RandomAdjustSharpness(sharpness_factor=2),
                        T.RandomAutocontrast(),
                        T.RandomInvert(0.25),
                        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                        T.RandomHorizontalFlip(),  # Random horizontal flip
                        T.RandomEqualize(),
                        T.ToTensor(),              # Convert image to tensor
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
                        ])

    ##define train and test datasets
    valdata = SiameseCustomImageDataFromFolder(test_csv, test_dir, train_transforms)

    ##define train and test iterators
    valloader = DataLoader(valdata, shuffle=False, pin_memory=False, batch_size=16)


    for i, data in enumerate(valloader, 0):
        if i < 1:
            x, _, label = data

            for j, img in enumerate(x):
                img = torch.squeeze(img).permute(1, 2, 0).detach().cpu().numpy()
                img = (255.0 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
                img = Image.fromarray(img)
                img.save('example_transforms_' + str(j) + '.png')



        # ##convert image prior to torch tensor
        # image = (255.0 * (x - x.min()) / (x.max() - x.min())).to(torch.uint8)
        # image = image[:3, ...]


        # ##plot image
        # fig, axes = plt.subplots(1, 1)
        # axes.imshow(image.permute(1, 2, 0))
        # axes.set_title('Example of Transformed Images')
        # plt.savefig('output.png')

    gc.collect()
