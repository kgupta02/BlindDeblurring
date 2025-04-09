# io_utils.py
# Encapsulates all reading/writing of images and other data

import os
import numpy as np
from skimage import io, color, img_as_ubyte

def read_image(path, as_gray=True):
    """
    Read an image from a given path and return it as a float32 NumPy array in the range [0,1]
    """
    image = io.imread(path)
    if as_gray and image.ndim == 3:
        image = color.rgb2gray(image)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image

def save_image(path, image):
    """
    Save a float32 NumPy array image to disk as an 8-bit PNG in [0,255].

    Args:
        path (str): Output file path
        image (np.ndarray): Image array
    """
    image = np.clip(image, 0, 1)
    image_ubyte = img_as_ubyte(image) # Convert to 8-bit
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.imsave(path, image_ubyte)

def save_kernel(path, kernel):
    """
    Save a blur kernel as an image, scale to visualize kernel

    Args:
        path (str): Output file path
        kernel (np.ndarray): blur kernel
    """
    if kernel.max() > 0:
        kernel_vis = kernel / kernel.max()
    else:
        kernel_vis = kernel
    kernel_vis = np.clip(kernel_vis, 0, 1)
    # Save kernel as image
    kernel_ubyte = img_as_ubyte(kernel_vis)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.imsave(path, kernel_ubyte)