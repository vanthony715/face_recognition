# -*- coding: utf-8 -*-
"""
@author avasque1@jh.edu
"""

import os, gc, time
from utils.utils import train_clf, val_clf
from simple_cnn import SimpleCNN
from custom_image_dset_from_folder import split_indices, CustomImageDataset

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
plt.ion()

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

gc.collect()
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    ##define paths
    datapath ='C:/Users/vanth/DnHFaces/open_data_set/classification_dset/'
    classes = os.listdir(datapath)
    numeric_labels = np.arange(0, len(classes))
    num_classes = len(classes)

    ##define image transforms
    # Define transformations for the training set
    train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(64),  # Randomly resize and crop to 64x64
                    transforms.RandomAdjustSharpness(sharpness_factor=2),
                    transforms.RandomAutocontrast(),
                    transforms.RandomHorizontalFlip(),  # Random horizontal flip
                    transforms.ToTensor(),              # Convert image to tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
                    ])

    # Define transformations for the validation and test sets
    test_val_transforms = transforms.Compose([
                    transforms.Resize(64),             # Resize to 64
                    transforms.CenterCrop(64),         # Crop to 64 from the center
                    transforms.ToTensor(),             # Convert image to tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
                    ])

    ##define train, test, valid datasets
    full_dataset = CustomImageDataset(root_dir=datapath)
    train_idx, val_idx, test_idx = split_indices(len(full_dataset))
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    ##apply transforms to dataset
    train_dataset.dataset = CustomImageDataset(root_dir=datapath, transform=train_transforms)
    val_dataset.dataset = CustomImageDataset(root_dir=datapath, transform=test_val_transforms)
    test_dataset.dataset = CustomImageDataset(root_dir=datapath, transform=test_val_transforms)

    ##define train, test, valid datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False)


    # Initialize the model, loss criterion, and optimizer
    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    ##train and validate
    model, train_loss, hist = train_clf(model, train_loader, val_loader, criterion,
                             optimizer, num_epochs=300)
    val_loss, y_true_val, y_pred_val, probs_val = val_clf(model, val_loader, criterion)
    test_loss, y_true_test, y_pred_test, probs_val = val_clf(model, test_loader, criterion)

    print('\nSaving Model...')
    torch.save(model.state_dict(), 'face_recognition_clf.pt')

    ##plot train loss
    fig, axes = plt.subplots(1,1)
    axes.plot(gaussian_filter1d(train_loss, sigma=10))
    axes.set_title('Training Loss Curve')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    plt.savefig('./train_loss.png')

    ##plot val metrics
    fig, axes = plt.subplots(1,1)
    axes.plot(hist['epoch'], gaussian_filter1d(hist['prec'], sigma=10), label='Precision Curve')
    axes.plot(hist['epoch'], gaussian_filter1d(hist['rec'], sigma=10), label='Recall Curve')
    axes.plot(hist['epoch'], gaussian_filter1d(hist['avg_loss'], sigma=10), label='Avg. Loss')
    axes.set_title('Validation Metrics')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Metric')
    axes.legend(loc='best')
    plt.savefig('./val_metrics.png')

    ##plot prec confidence curve
    fig, axes = plt.subplots(1,1)
    axes.plot(hist['avg_prob'], gaussian_filter1d(hist['prec'], sigma=10), label='Precision Confidence Curve')
    axes.set_title('Precision-Confidence Curve')
    axes.set_xlabel('Confidence')
    axes.set_ylabel('Precision')
    plt.savefig('./prec_conf_curve.png')

    ##plot recall confidence curve
    fig, axes = plt.subplots(1,1)
    axes.plot(hist['avg_prob'], gaussian_filter1d(hist['rec'], sigma=10), label='Recall Confidence Curve')
    axes.set_title('Recall-Confidence Curve')
    axes.set_xlabel('Confidence')
    axes.set_ylabel('Recall')
    plt.savefig('./recall_conf_curve.png')

    print('\n--------Val Summary Metrics---------')
    print('Precision: ', precision_score(y_true_val, y_pred_val, average='weighted'))
    print('Recall: ', recall_score(y_true_val, y_pred_val, average='weighted'))
    print('Avg. Loss: ', val_loss)

    print('\n--------Test Summary Metrics---------')
    print('Precision: ', precision_score(y_true_test, y_pred_test, average='weighted'))
    print('Recall: ', recall_score(y_true_test, y_pred_test, average='weighted'))
    print('Avg. Loss: ', test_loss)
    print('\nTest CM')
    cm = confusion_matrix(y_true_test, y_pred_test, labels=numeric_labels)
    print(cm)
    fig, axes = plt.subplots(1, 1)
    class_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5,
                 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10,}
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_map)
    disp.plot(cmap=plt.cm.Blues, colorbar=False).figure_.savefig('confusion_matrix.png')
    # plt.savefig('test_cm.png')
