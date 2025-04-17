import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import cg, LinearOperator
from blur_model import compute_mask
from priors import compute_gradients


def hyper_laplacian_grad(I):
    grad_x, grad_y = compute_gradients(I)
    return np.sign(grad_x) * ((np.abs(grad_x) + 1e-8) ** -0.2), np.sign(grad_y) * ((np.abs(grad_y) + 1e-8) ** -0.2)

def update_I(I, B, K, M, lam):
    I_conv = convolve2d(I, K, mode='same')
    K_flip = np.flip(np.flip(K, axis=0), axis=1)
    numerator = B / np.maximum(I_conv, 1e-8) - M + 1
    update = convolve2d(numerator, K_flip, mode='same')
    grad_x, grad_y = hyper_laplacian_grad(I)
    regularizer = lam * (grad_x + grad_y)
    return I * update / (1 + regularizer)

def update_K(B, I, K_prev, M, beta, kernel_shape=(15, 15)):

    grad_Bx, grad_By = compute_gradients(B)
    grad_Ix, grad_Iy = compute_gradients(I)

    I_conv = convolve2d(I, K_prev, mode='same', boundary='symm')
    W_pix = 1.0 / np.maximum(M * I_conv, 1e-8)

    kh, kw = kernel_shape
    H, W = I.shape
    cy, cx = H // 2, W // 2
    patch_Ix = grad_Ix[cy - kh // 2 : cy - kh // 2 + kh,
                       cx - kw // 2 : cx - kw // 2 + kw]
    patch_Iy = grad_Iy[cy - kh // 2 : cy - kh // 2 + kh,
                       cx - kw // 2 : cx - kw // 2 + kw]
    flipped_grad_Ix = np.flip(np.flip(patch_Ix, axis=0), axis=1)
    flipped_grad_Iy = np.flip(np.flip(patch_Iy, axis=0), axis=1)

    kh, kw = kernel_shape

    def A_operator(K_flat):
        K = K_flat.reshape(kernel_shape)
        conv_Ix = convolve2d(grad_Ix, K, mode='same', boundary='symm')
        conv_Iy = convolve2d(grad_Iy, K, mode='same', boundary='symm')
        Wx = W_pix * conv_Ix
        Wy = W_pix * conv_Iy

        Wx_patch = Wx[cy - kh // 2 : cy - kh // 2 + kh,
                       cx - kw // 2 : cx - kw // 2 + kw]
        Wy_patch = Wy[cy - kh // 2 : cy - kh // 2 + kh,
                       cx - kw // 2 : cx - kw // 2 + kw]

        term_x = convolve2d(Wx_patch, flipped_grad_Ix, mode='valid', boundary='symm')
        term_y = convolve2d(Wy_patch, flipped_grad_Iy, mode='valid', boundary='symm')
        return (term_x + term_y + beta * K).ravel()

    op = LinearOperator(shape=(kh * kw, kh * kw), matvec=A_operator, dtype=np.float32)

    b_x = convolve2d(W_pix * grad_Bx, flipped_grad_Ix, mode='valid', boundary='symm')
    b_y = convolve2d(W_pix * grad_By, flipped_grad_Iy, mode='valid', boundary='symm')
    b_total = b_x + b_y

    H_b, W_b = b_total.shape
    cy_b, cx_b = H_b // 2, W_b // 2
    b_crop = b_total[cy_b - kh // 2 : cy_b - kh // 2 + kh,
                     cx_b - kw // 2 : cx_b - kw // 2 + kw]
    b = b_crop.ravel()

    K0 = np.ones(kernel_shape) / (kh * kw)
    K_flat, _ = cg(op, b, x0=K0.ravel(), maxiter=50)
    K = K_flat.reshape(kernel_shape)
    K[K < 0] = 0
    s = K.sum()
    K = K / s if s > 1e-8 else np.ones(kernel_shape) / (kh * kw)
    return K

def update_latent_image(I, K, B, latent_map, lambda_grad):
    """
    Update the latent image using Richardson-Lucy or similar optimization method.
    """
    updated_image = I * ( (B - latent_map * convolve2d(I, K, mode='same')) / (latent_map * convolve2d(I, K, mode='same')) )
    updated_image = updated_image + lambda_grad * np.gradient(updated_image)
    
    return updated_image

def update_kernel(I, B, latent_map, beta_kernel):
    """
    Update the blur kernel using a least squares optimization approach.
    """
    # Adjust the kernel estimation based on the weighted least squares
    kernel_update = np.sum(latent_map * (B - convolve2d(I, K, mode='same')))  # Weighted error term
    kernel_update = kernel_update + beta_kernel * np.sum(K**2)  # Regularization

    return kernel_update

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
