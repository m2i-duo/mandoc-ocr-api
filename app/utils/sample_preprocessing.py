from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2
from config import config


def preprocess(img):
    """
    Scale the input image to the desired dimensions, transpose it for TensorFlow,
    and normalize the grayscale values.
    """
    # Augment the dataset by applying random horizontal stretches to the images
    if config.AUGMENT_IMAGE:
        stretch = random.uniform(-0.5, 0.5)  # Random factor between -0.5 and +0.5
        w_stretched = max(int(img.shape[1] * (1 + stretch)), 1)  # Ensure at least width of 1
        img = cv2.resize(img, (w_stretched, img.shape[0]))

    # Calculate scaling factor to fit the image within the target dimensions
    (h, w) = img.shape
    fx = w / config.IMAGE_WIDTH
    fy = h / config.IMAGE_HEIGHT
    f = max(fx, fy)  # Scale according to the larger dimension

    # Compute the new size for the image
    new_size = (
        max(min(config.IMAGE_WIDTH, int(w / f)), 1),
        max(min(config.IMAGE_HEIGHT, int(h / f)), 1)
    )
    img = cv2.resize(img, new_size)

    # Create a blank target image and place the resized image within it
    target = np.ones([config.IMAGE_HEIGHT, config.IMAGE_WIDTH]) * 255  # Initialize with white
    target[0:new_size[1], 0:new_size[0]] = img

    # Transpose the image for TensorFlow compatibility
    img = cv2.transpose(target)

    # Normalize the image to zero mean and unit variance
    (mean, std_dev) = cv2.meanStdDev(img)
    mean = mean[0][0]
    std_dev = std_dev[0][0]
    img = img - mean
    img = img / std_dev if std_dev > 0 else img

    return img


def convert_to_binary(im, threshold=128):
    """
    Convert an image to binary based on a given threshold.
    Pixels below the threshold are set to 0, and those above are set to 255.
    """
    im[im < threshold] = 0
    im[im >= threshold] = 255
    return im
