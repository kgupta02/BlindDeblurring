from scipy.signal import convolve2d
import numpy as np

def blur_model(image, kernel):
    """
    blur the image as the model which the paper proposed

    Args:
        image (np.ndarray): Image array
        kernel (np.ndarray): Kernel array
    """
    blur_image = convolve2d(image, kernel, model='same')
    mask = np.where(blur_image <= 1, 1, 1 / blur_image)
    blur_image = blur_image * mask

    return blur_image