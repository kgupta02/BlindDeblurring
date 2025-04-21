# blur_model.py
from scipy.signal import convolve2d
import numpy as np

def blur_model(image, kernel):
    """
    blur the image as the model which the paper proposed

    Args:
        image (np.ndarray): Image array
        kernel (np.ndarray): Kernel array
    """
    blur_image = convolve2d(image, kernel, mode='same')
    latent_map = np.where(blur_image <= 1, 1, 1 / blur_image)
    blur_image = latent_map * blur_image

    return blur_image

def compute_mask(I_blurred, K):
    convolved = convolve2d(I_blurred, K, mode='same')
    mask = np.where(convolved <= 1, 1, 1 / convolved)
    return mask