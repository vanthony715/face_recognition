# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""
import os
from PIL import Image
import numpy as np

def create_mosaic(image_paths, rows, cols):
    # Load images
    images = [Image.open(path) for path in image_paths]
    images = [i.resize((64, 64)) for i in images]

    # Assume all images are the same size
    width, height = images[0].size

    # Create a blank canvas for the mosaic
    mosaic = Image.new('RGB', (cols * width, rows * height))

    # Paste images into the mosaic
    for idx, image in enumerate(images):

        x = idx % cols * width
        y = idx // cols * height
        mosaic.paste(image, (x, y))

    return mosaic

if __name__== "__main__":
    ##define datapath
    datapath1 = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/"
    datapath2 = "cv_deep_learning/final/figures/siamese/examples_for_mosaic/example_transforms/"
    datapath0 = datapath1 + datapath2

    # List of image paths
    image_paths = os.listdir(datapath0)
    image_paths = [datapath0 + i for i in image_paths]
    rows, cols = 4, 4  # Example for a 2x2 grid

    # Create and show the mosaic
    mosaic = create_mosaic(image_paths, rows, cols)
    mosaic.show()  # This will open the mosaic image
    mosaic.save('./transforms_mosaic.png')
