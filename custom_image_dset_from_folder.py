# -*- coding: utf-8 -*-
"""
@author avasque1@jh.edu
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label, classname in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, classname)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def split_indices(n, val_pct=0.15, test_pct=0.15, shuffle=True):
    """ Split indices into training, validation, and test datasets """
    indices = list(range(n))
    if shuffle:
        np.random.shuffle(indices)

    test_size = int(n * test_pct)
    val_size = int(n * val_pct)
    train_indices = indices[:-test_size - val_size]
    val_indices = indices[len(train_indices):len(train_indices) + val_size]
    test_indices = indices[len(train_indices) + val_size:]

    return train_indices, val_indices, test_indices

# Example usage (this should be commented out in the final code)
# dataset = CustomImageDataset(root_dir='path/to/data')
# train_idx, val_idx, test_idx = split_indices(len(dataset))
# train_dataset = Subset(dataset, train_idx)
# val_dataset = Subset(dataset, val_idx)
# test_dataset = Subset(dataset, test_idx)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
