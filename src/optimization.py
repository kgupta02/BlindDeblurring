import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import cg
from blur_model import compute_mask
from priors import compute_gradients


def hyper_laplacian_grad(I):
    #grad_x = np.diff(I, axis=1, append=I[:, -1:])
    #grad_y = np.diff(I, axis=0, append=I[-1:, :])
    grad_x, grad_y = compute_gradients(I)
    return np.sign(grad_x) * (np.abs(grad_x) ** -0.2), np.sign(grad_y) * (np.abs(grad_y) ** -0.2)

def update_I(I, B, K, M, lam):
    I_conv = convolve2d(I, K, mode='same')
    K_flip = np.flip(np.flip(K, axis=0), axis=1)
    numerator = B / np.maximum(I_conv, 1e-8) - M + 1
    update = convolve2d(numerator, K_flip, mode='same')
    grad_x, grad_y = hyper_laplacian_grad(I)
    regularizer = lam * (grad_x + grad_y)
    return I * update / (1 + regularizer)

def update_K(B, I, M, K, beta, kernel_shape=(15, 15)):
    grad_Bx, grad_By = compute_gradients(B)
    grad_Ix, grad_Iy = compute_gradients(I)

    I_conv = convolve2d(I, K, mode='same')
    W = 1.0 / np.maximum(K * I_conv, 1e-8)

    def A_operator(K_flat):
        K = K_flat.reshape(kernel_shape)
        conv_Ix = convolve2d(grad_Ix, K, mode='same')
        conv_Iy = convolve2d(grad_Iy, K, mode='same')
        Wx = W * conv_Ix
        Wy = W * conv_Iy
        term_x = convolve2d(Wx, np.flip(np.flip(grad_Ix, axis=0), axis=1), mode='same')
        term_y = convolve2d(Wy, np.flip(np.flip(grad_Iy, axis=0), axis=1), mode='same')
        return (term_x + term_y + beta * K).ravel()

    b_x = convolve2d(W * grad_Bx, np.flip(np.flip(grad_Ix, axis=0), axis=1), mode='same')
    b_y = convolve2d(W * grad_By, np.flip(np.flip(grad_Iy, axis=0), axis=1), mode='same')
    b = b_x + b_y

    K0 = np.ones(kernel_shape) / np.prod(kernel_shape)
    K_flat, _ = cg(A_operator, b.ravel(), x0=K0.ravel(), maxiter=50)

    K = K_flat.reshape(kernel_shape)
    K[K < 0] = 0
    K /= K.sum()
    return K

def optimize(B, K_init, lam=0.008, beta=2, tmax=50, jmax=4):
    I = B.copy()
    K = K_init.copy()
    for j in range(jmax):
        for t in range(tmax):
            M = compute_mask(I, K)
            I = update_I(I, B, K, M, lam)
        M = compute_mask(I, K)
        K = update_K(B, I, K, M, beta)
    return I, K
