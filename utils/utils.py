# -*- coding: utf-8 -*-
"""
@author avasque1@jh.edu
"""

import os, shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def make_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        pass
    os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training loop
def train_clf(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.train()  # Set the model to training mode
    hist = {'epoch': [], "prec": [], "rec": [], "avg_prob": [],"avg_loss": []}
    train_loss = []
    for epoch in range(num_epochs):
        for images, labels in tqdm(train_loader, desc='Training'):
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_loss.append(loss.item())

        if epoch%3==0:
            avg_loss, y_true, y_pred, probs = val_clf(model, val_loader, criterion)
            hist['epoch'].append(epoch)
            hist['prec'].append(precision_score(y_true, y_pred, average='weighted'))
            hist['rec'].append(recall_score(y_true, y_pred, average='weighted'))
            hist['avg_prob'].append(torch.mean(torch.tensor(probs)).item())
            hist['avg_loss'].append(avg_loss)
    return model, train_loss, hist

# Validation loop
def val_clf(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():  # No need to track gradients
        for images, labels in tqdm(val_loader, desc='Testing'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            y_true.append(labels.item())
            _, pred = torch.max(outputs.data, 1)
            probs.append(torch.max(nn.functional.softmax(outputs, dim=1)).item())
            y_pred.append(pred.item())
    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss, y_true, y_pred, probs

def val_siam(model, testloader, pass_thresh, criterion, device):
    #test the network
    total_loss = 0
    y_true, y_pred, distances = [], [], []
    with torch.no_grad():  # No need to track gradients
        for i, data in enumerate(testloader, 0):
            img1, img2, label = data

            out1, out2 = model(img1.to(device), img2.to(device))
            loss_contrastive = criterion(out1.to(device), out2.to(device),
                                         label.to(device))
            total_loss += loss_contrastive.item()
            euclid_dist = F.pairwise_distance(out1, out2)

            ##record
            y_true.append(label.item())
            distances.append(euclid_dist.item())

            ##has to be less then specified distance to be predicted genuine
            # print('\nEuclid Distance: ', euclid_dist.item())
            # print('Pass Thresh: ', pass_thresh)
            if euclid_dist.item() <= pass_thresh:
                y_pred.append(0)
            else:
                y_pred.append(1)

    avg_loss = total_loss / len(testloader)

    return avg_loss, y_true, y_pred, distances

def train_siam(network, device, trainloader, testloader, num_epochs, max_lr, criterion, optimizer):
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    ##train
    train_loss = []
    pass_thresh = 0.21
    hist = {'epoch': [], "prec": [], "rec": [], "dist": [],"avg_loss": []}
    for epoch in range(num_epochs):
        for idx, data in tqdm(enumerate(trainloader), desc='Training'):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            out1, out2 = network(img0, img1)
            loss_contrastive = criterion(out1, out2, label)
            loss_contrastive.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_contrastive.item():.4f}')
        train_loss.append(loss_contrastive.item())

        ##step lr scheduler
        sched.step()
        print('Learning Rate: ', optimizer.param_groups[0]["lr"])
        if epoch % 2 == 0:

            avg_loss, y_true, y_pred, distances = val_siam(network, testloader,
                                                           pass_thresh, criterion, device)
            hist['epoch'].append(epoch)
            hist['prec'].append(precision_score(y_true, y_pred, average='weighted'))
            hist['rec'].append(recall_score(y_true, y_pred, average='weighted'))
            hist['dist'].append(torch.mean(torch.tensor(distances)).item())
            hist['avg_loss'].append(avg_loss)

    return network, train_loss, hist

def visualize_siam_example(img1, img2, score, threshold, truth_label, plot_save_name):
    pair_map = {0: 'Genuine', 1: 'imposter'}
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(torch.squeeze(img1).permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
    axes[0].set_title('Sample 1')
    axes[1].imshow(torch.squeeze(img2).permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
    axes[1].set_title('Sample 2')
    if int(truth_label.item()) == 0 and score.item() <= threshold:
        correct = 'Yes'
    elif int(truth_label.item()) == 1 and score.item() > threshold:
        correct = 'Yes'
    else:
        correct = 'No'
    fig.suptitle('Truth Label: ' + str(pair_map[int(truth_label.item())]) + ' - Euclid Dist: ' + str(np.round(score.item(), 4)) + ' - Correct: ' + correct)
    fig.tight_layout()
    plt.savefig(plot_save_name)

def create_pairs(data, num_per_pair, num_pairs, writepath):
    modes = [0, 1] #mode 0 is genuine pairs
    sdict = {'gen': [], 'imp': [], 'class': []} ##record to use during training
    for c, clss in tqdm(enumerate(sorted(data['class'].unique())), desc='Creating Pairs'):
        for pair in range(num_pairs):
            for mode in modes:
                if mode == 1:
                    ##define writepaths
                    write_gen_pair = writepath + str(pair) + '_' + clss + '_gen/'
                    write_imp_pair = writepath + str(pair) + '_' + clss + '_imp/'

                    ##create writepaths
                    make_dir(write_gen_pair)
                    make_dir(write_imp_pair)

                    ##write genuine pairs
                    gdf = data.loc[data['class'] == clss]
                    sample_df = gdf.sample(n=num_per_pair)
                    for i, path in enumerate(sample_df.filepath):
                        shutil.copy(path, write_gen_pair + str(pair) + '_' + str(i) + '.png')
                        sdict['gen'].append(write_gen_pair + str(pair) + '_' + str(i) + '.png')

                    ##write imposter pair
                    idf = data.loc[data['class'] != clss]
                    sample_df = idf.sample(n=num_per_pair)
                    for i, path in enumerate(sample_df.filepath):
                        shutil.copy(path, write_imp_pair + '' + str(pair) + '_' + str(i) + '.png')
                        sdict['imp'].append(write_imp_pair + str(pair) + '_' + str(i) + '.png')
                        sdict['class'].append(mode)
                else:
                    ##define writepaths
                    write_gen1_pair = writepath + str(pair) + '_' + clss + '_gen1/'
                    write_gen2_pair = writepath + str(pair) + '_' + clss + '_gen2/'

                    ##create writepaths
                    make_dir(write_gen1_pair)
                    make_dir(write_gen2_pair)

                    ##write genuine pairs
                    gdf = data.loc[data['class'] == clss]
                    sample_df = gdf.sample(n=num_per_pair)
                    for i, path in enumerate(sample_df.filepath):
                        shutil.copy(path, write_gen1_pair + str(pair) + '_' + str(i) + '.png')
                        sdict['gen'].append(write_gen1_pair + str(pair) + '_' + str(i) + '.png')

                    ##write imposter pair
                    gdf = data.loc[data['class'] == clss]
                    sample_df = gdf.sample(n=num_per_pair)
                    for i, path in enumerate(sample_df.filepath):
                        shutil.copy(path, write_gen2_pair + str(pair) + '_' + str(i) + '.png')
                        sdict['imp'].append(write_gen2_pair + str(pair) + '_' + str(i) + '.png')
                        sdict['class'].append(mode)

    return pd.DataFrame(sdict)
