# -*- coding: utf-8 -*-
"""
@author avasque1@jh.edu
"""

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
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
        self.fc = nn.Sequential(nn.Linear(8192, num_classes))

    def forward(self, x):
        '''
        Only one item forward pass
        '''
        x = self.cnn(x) #convolution
        x = x.view(x.size()[0], -1)
        x = self.fc(x) #fully connected
        return x
