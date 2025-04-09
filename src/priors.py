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


def compute_divergence(grad_h, grad_v):
    '''
    Compute the divergence of the gradient field (negative adjoint)
    
    Returns:
        div (np.ndarray): The divergence (same shape as input)
    '''
    div = np.zeros_like(grad_h)
    
    div[:, 0] = grad_h[:, 0]
    div[:, 1:] += grad_h[:, 1:] - grad_h[:, :-1]
    
    div[0, :] += grad_v[0, :]
    div[1:, :] += grad_v[1:, :] - grad_v[:-1, :]
    
    return div


def hyper_laplacian_prior(I, alpha=0.8):
    '''
    P(I) = sum(|∇_h I|^α + |∇_v I|^α)
    
    Args:
        alpha (float): exponent of the hyper-Laplacian

    Returns:
        float: the summed prior cost
    '''
    grad_h, grad_v = compute_gradients(I)
    return np.sum(np.abs(grad_h)**alpha + np.abs(grad_v)**alpha)


def grad_hyper_laplacian_prior(I, alpha=0.8, eps=1e-8):
    '''
    Computes the gradient of the hyper-laplacian prior with respect to I
    '''
    grad_h, grad_v = compute_gradients(I)
    
    # Force terms for each gradient direction
    force_h = alpha * np.sign(grad_h) * (np.abs(grad_h) + eps)**(alpha - 1)
    force_v = alpha * np.sign(grad_v) * (np.abs(grad_v) + eps)**(alpha - 1)
    
    grad_prior = compute_divergence(force_h, force_v)
    
    return grad_prior

def kernel_l2_prior(K):
    """    
    P(K) = ||K||^2
    """
    return np.sum(K**2)


def grad_kernel_l2(K):
    """
    d/dK P(K) = 2 * K
    """
    return 2 * K