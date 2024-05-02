# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

import os
import scipy
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.utils import visualize_siam_example
from utils.utils import count_parameters, train_siam, val_siam
from simple_siam import SimpleSiam, ContrastiveLoss
from siamese_custom_image_data_from_folder import SiameseCustomImageDataFromFolder

import matplotlib.pyplot as plt
plt.ioff()

import gc
gc.collect()

import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()


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
    train_transforms = v2.Compose([
                        v2.RandomResizedCrop(64),  # Randomly resize and crop to 64x64
                        v2.RandomAdjustSharpness(sharpness_factor=2),
                        v2.RandomAutocontrast(),
                        v2.RandomInvert(0.25),
                        v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                        v2.RandomHorizontalFlip(),  # Random horizontal flip
                        v2.RandomEqualize(),
                        v2.ToTensor(),              # Convert image to tensor
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
                        ])

    # Define transformations for the validation and test sets
    test_val_transforms = v2.Compose([
                    v2.Resize(64),             # Resize to 64
                    v2.CenterCrop(64),         # Crop to 64 from the center
                    v2.ToTensor(),             # Convert image to tensor
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
                    ])

    ##hyperparameters
    epochs = 200
    lr = 0.00001 #starting learning rate
    pass_thresh = 0.21 #pass threshold to be considered genuine
    margin = 0.6 #training margin error
    num_vis_examples = 10 #number of pairs to visualize with the trained model

    ##define train and test datasets
    traindata = SiameseCustomImageDataFromFolder(train_csv, train_dir, transform=train_transforms)
    testdata = SiameseCustomImageDataFromFolder(test_csv, test_dir, transform=test_val_transforms)

    ##define train and test iterators
    trainloader = DataLoader(traindata, shuffle=True, pin_memory=False, batch_size=256)
    testloader = DataLoader(testdata, shuffle=False, pin_memory=False, batch_size=1)

    ##Initialize Network
    network = SimpleSiam().to(device)
    print(network)

    ##initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr)
    print('\nNum Trainable Parameters: ', count_parameters(network), '\n')

    ##initialize loss function
    criterion = ContrastiveLoss(margin=margin)

    ##fit model to data and visualize training losses
    print('\n----------Training----------')
    model, train_loss, hist = train_siam(network, device, trainloader, testloader,
                                         epochs, lr, criterion, optimizer)
    torch.save(model.state_dict(), 'siam' + str(margin) + '.pt') #save weights

    ##plot train Loss
    fig, axes = plt.subplots(1, 1)
    axes.plot(scipy.ndimage.gaussian_filter1d(train_loss, sigma=50)[50:])
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Contrative Loss')
    axes.set_title('Contrastive Loss vs. Iteration')
    plt.grid()
    plt.tight_layout()

    ##test the model
    model = SimpleSiam().to(device) #load net architecture
    model.load_state_dict(torch.load('siam' + str(margin) + '.pt')) #load trained model state
    model.eval() #set grad parameters to no grad updates
    val_loss, y_true, y_pred, distances = val_siam(model, testloader, pass_thresh,
                                                   criterion, device) #get performance

    ##visualize
    for i in range(num_vis_examples - 1):
        img1, img2, label = next(iter(testloader))
        img1, img2 = img1.to(device), img2.to(device)
        out1, out2 = network(img1, img2)
        euclid_dist = F.pairwise_distance(out1, out2, keepdim=True)
        visualize_siam_example(img1, img2, score=euclid_dist,
                               threshold=pass_thresh,
                               truth_label=label,
                               plot_save_name='example_' + str(i) + '.png')

    ##plot train loss
    fig, axes = plt.subplots(1,1)
    axes.plot(gaussian_filter1d(train_loss, sigma=2))
    axes.set_title('Training Contrastive Loss Curve')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    plt.savefig('./train_loss.png')

    ##plot Recall Euclid curve
    fig, axes = plt.subplots(1,1)
    axes.plot(hist['epoch'], gaussian_filter1d(hist['dist'], sigma=2),
              label='Euclid-Distance Curve')
    axes.set_title('Epoch vs Euclid Distance')
    axes.set_ylabel('Euclidean Distance')
    axes.set_xlabel('Epoch')
    plt.savefig('./epoch_eucdist.png')

    ##plot val metrics on two different y-axes
    fig, axes = plt.subplots(1,1)
    axes.plot(hist['epoch'], gaussian_filter1d(hist['prec'], sigma=2),
              c='blue', label='Precision Curve')
    axes.plot(hist['epoch'], gaussian_filter1d(hist['rec'], sigma=2), c='green', label='Recall Curve')
    axes2 = axes.twinx()
    axes2.plot(hist['epoch'], gaussian_filter1d(hist['avg_loss'], sigma=2),
               label='Avg. Contrastive Loss', c='brown')
    axes2.set_ylabel('Contrastive Loss')
    # axes2.legend(loc='best')
    axes.set_title('Validation Metrics')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Metric')
    axes.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./val_metrics.png')

    print('\n--------Val Summary Metrics---------')
    print('Precision: ', precision_score(y_true, y_pred, average='weighted'))
    print('Recall: ', recall_score(y_true, y_pred, average='weighted'))
    print('Avg. Loss: ', val_loss)

    numeric_labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=numeric_labels)

    print('CM: \n')
    print(cm)

    ##plot confusion matrix
    fig, axes = plt.subplots(1, 1)
    class_map = {'genuine': 0, 'imposter': 1}
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_map)
    disp.plot(cmap=plt.cm.Blues, colorbar=False).figure_.savefig('confusion_matrix.png')

    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['eucdist'] = distances
    #define pass criteria
    gen_correct = len(df[(df.y_true == 0) & (df.eucdist <= 0.21)])
    imp_correct = len(df[(df.y_true == 1) & (df.eucdist > 0.21)])


    print('\n----------Test Performance----------')
    print('Number of test samples: ', len(df))
    print('Number Genuine Pairs Correctly Predicted: ', gen_correct)
    print('Number Imposter Pairs Correctly Predicted: ', imp_correct)
    print('Accuracy: ', np.round((gen_correct + imp_correct)/len(df), 2))
    print('\n')
