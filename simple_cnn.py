# -*- coding: utf-8 -*-
"""
@author avasque1@jh.edu
"""

import torch.nn as nn

# Example: A simple CNN for classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):  # Assuming 10 classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 input channels, 16 output channels
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # Reduce size by 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8192, num_classes)  # Adjust size according to your final feature map dimensions

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x)
        return x
