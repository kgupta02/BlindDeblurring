# priors.py
'''
GOAL - To provide implementations for:
    - Computing image gradients
    - Evaluating hyper-Laplacian prior on an image and corresponding gradient
    - Evaluating L2 prior for blur kernels
'''
import numpy as np

def compute_gradients(I):
    '''
    Compute forward differences as approximations of the horizontal and vertical gradients
    
    Args:
        I (np.ndarray): 2D input image with normalized values [0,1]
    
    Returns:
        grad_h, grad_v (np.ndarray): Horizontal and vertical gradients
    '''
    grad_h = np.zeros_like(I)
    grad_v = np.zeros_like(I)

    grad_h[:, :-1] = I[:, 1:] - I[:, :-1]
    grad_v[:-1, :] = I[1:, :] - I[:-1, :]

    return grad_h, grad_v


