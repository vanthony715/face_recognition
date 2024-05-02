# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""
import torch
import torch.nn as nn
from torchviz import make_dot

import matplotlib.pyplot as plt

from simple_siam import SimpleSiam
from simple_cnn import SimpleCNN

if __name__ == "__main__":
    ##test the model
    # modelpath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/cv_deep_learning/final/siam0.6.pt"
    modelpath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/cv_deep_learning/final/face_recognition_clf_300_epochs.pt"
    # model = SimpleSiam() #load net architecture
    model = SimpleCNN(num_classes=11)
    model.load_state_dict(torch.load(modelpath)) #load trained model state

    ##siamese network make random example and forward pass
    # x1 = torch.randn(1, 3, 64, 64)
    # x2 = torch.randn(1, 3, 64, 64)
    # out = model(x1, x2) ##for siam network

    ##classifier network make random example and forward pass
    x = torch.randn(1, 3, 64, 64)
    out = model(x) ##for siam network

    ##siamese network
    # dot = make_dot(out, params=dict(list(model.named_parameters()) + [('input', x1)]))
    # dot.render('siam_net_graph', format='png')  # This saves the graph as a PNG file

    ##classifier network
    dot = make_dot(out, params=dict(list(model.named_parameters()) + [('input', x)]))
    dot.render('clf_net_graph.png', format='png')  # This saves the graph as a PNG file
