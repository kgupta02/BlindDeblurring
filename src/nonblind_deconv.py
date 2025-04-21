# nonblind_deconv.py
import numpy as np
from scipy.signal import convolve2d

def nonblind_deconv(B, K, lambda_val=0.008, num_iter=50, epsilon=1e-8):    
    """
    Implements a non-blind deconvolution method, recovers the final latent (sharp) image
    """
    I = B.copy()
    
    K_flip = np.flipud(np.fliplr(K))
    
    for t in range(num_iter):
        I_conv = convolve2d(I, K, mode='same', boundary='symm')
        
        M = np.ones_like(I_conv)
        mask = I_conv > 1
        M[mask] = 1.0 / (I_conv[mask] + epsilon)
        
        ratio_input = (B / (I_conv + epsilon)) - M + 1
        
        ratio_term = convolve2d(ratio_input, K_flip, mode='same', boundary='symm')
        
        grad_v, grad_h = np.gradient(I)
        
        grad_h_safe = np.maximum(np.abs(grad_h), epsilon)
        grad_v_safe = np.maximum(np.abs(grad_v), epsilon)
        
        P_prime_h = np.sign(grad_h) * (grad_h_safe ** (-0.2))
        P_prime_v = np.sign(grad_v) * (grad_v_safe ** (-0.2))
        P_prime = P_prime_h + P_prime_v
        
        denom = 1.0 + lambda_val * P_prime
        
        I = I * (ratio_term / (denom + epsilon))
        
        I = np.clip(I, 0, None)
    
    return I