# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:45:40 2024

@author: vanth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSiam(nn.Module):
    def __init__(self):
        super(SimpleSiam, self).__init__()
        '''
        Simple Siamese Network mmodeled after SimpleCNN
        '''
        ##define convolution network
        self.cnn = nn.Sequential( ##modified VGG16
                                 nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 input channels, 16 output channels
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2),  # Reduce size by 2

                                 nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2)
                                 )

        ##define fully connected
        self.fc = nn.Sequential(nn.Linear(8192, 1),
                                nn.Sigmoid())

    def forward_once(self, x):
        '''
        Only one item forward pass
        '''
        x = self.cnn(x) #convolution
        x = x.view(x.size()[0], -1)
        x = self.fc(x) #fully connected
        return x

    def forward(self, x1, x2):
        '''
        Uses one item forward pass to forward pass pair
        '''
        x1 = self.forward_once(x1) #forward pass of img0
        x2 = self.forward_once(x2) #forward pass of img1
        return x1, x2

class ContrastiveLoss(nn.Module):
    '''
    Contrastive loss function
    '''
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin ##threshold error margin

    def forward(self, output1, output2, label):
        ##calculate euclidean distance
        euclid_dist = F.pairwise_distance(output1, output2, keepdim=True)

        ##calculate contrastive loss
        c_loss = torch.mean((1 - label) * torch.pow(euclid_dist, 4) +
                           (label) * torch.pow(torch.clamp(self.margin - euclid_dist,
                                                           min=0.0), 2))
        return c_loss
