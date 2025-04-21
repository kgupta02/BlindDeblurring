import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import LinearOperator
from pyramid import build_pyramid, upsample_kernel
from priors import compute_gradients

### --- Utility Functions --- ###

def hyper_laplacian_grad(I):
    grad_x, grad_y = compute_gradients(I)
    epsilon = 1e-8
    reg_x = np.sign(grad_x) * (np.abs(grad_x) + epsilon) ** -0.2
    reg_y = np.sign(grad_y) * (np.abs(grad_y) + epsilon) ** -0.2
    return reg_x, reg_y


def compute_latent_map(I, K):
    conv = convolve2d(I, K, mode='same', boundary='symm')
    M = np.where(conv <= 1, 1.0, 1.0 / np.maximum(conv, epsilon:=1e-8))
    return M


def custom_cg(A_op, b, x0, tol=1e-3, maxiter=100):
    x = x0.copy()
    r = b - A_op(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    denom_eps = 1e-8

    for i in range(maxiter):
        Ap = A_op(p)
        dot_pAp = np.dot(p, Ap)
        if abs(dot_pAp) < denom_eps:
            break
        alpha = rs_old / dot_pAp
        x_new = x + alpha * p
        rel_error = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + denom_eps)
        if rel_error < tol:
            x = x_new
            break
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if rs_old < denom_eps:
            x = x_new
            break
        p = r + (rs_new / (rs_old + denom_eps)) * p
        x = x_new
        rs_old = rs_new

    return x


def update_I(I, B, K, M, lam):
    I_conv = convolve2d(I, K, mode='same', boundary='symm')
    K_flip = np.flip(np.flip(K, axis=0), axis=1)
    correction = B / np.maximum(I_conv, 1e-8) - M + 1
    update = convolve2d(correction, K_flip, mode='same', boundary='symm')
    grad_x, grad_y = hyper_laplacian_grad(I)
    regularizer = lam * (grad_x + grad_y)
    return I * update / (1 + regularizer)


def update_K(B, I, K_prev, M, beta, kernel_shape=(15,15)):
    grad_Bx, grad_By = compute_gradients(B)
    grad_Ix, grad_Iy = compute_gradients(I)

    I_conv = convolve2d(I, K_prev, mode='same', boundary='symm')
    W_pix = 1.0 / np.maximum(M * I_conv, 1e-8)

    kh, kw = kernel_shape
    H, W = I.shape
    cy, cx = H//2, W//2
    patch_Ix = grad_Ix[cy-kh//2:cy-kh//2+kh, cx-kw//2:cx-kw//2+kw]
    patch_Iy = grad_Iy[cy-kh//2:cy-kh//2+kh, cx-kw//2:cx-kw//2+kw]
    flipped_Ix = np.flip(np.flip(patch_Ix, 0), 1)
    flipped_Iy = np.flip(np.flip(patch_Iy, 0), 1)

    def A_op(vec):
        K = vec.reshape(kernel_shape)
        cIx = convolve2d(grad_Ix, K, mode='same', boundary='symm')
        cIy = convolve2d(grad_Iy, K, mode='same', boundary='symm')
        Wx = W_pix * cIx
        Wy = W_pix * cIy
        Wx_p = Wx[cy-kh//2:cy-kh//2+kh, cx-kw//2:cx-kw//2+kw]
        Wy_p = Wy[cy-kh//2:cy-kh//2+kh, cx-kw//2:cx-kw//2+kw]
        tx = convolve2d(Wx_p, flipped_Ix, mode='valid', boundary='symm')
        ty = convolve2d(Wy_p, flipped_Iy, mode='valid', boundary='symm')
        return (tx + ty + beta * K).ravel()

    op = LinearOperator((kh*kw, kh*kw), matvec=A_op, dtype=np.float32)

    bx = convolve2d(W_pix*grad_Bx, flipped_Ix, mode='valid', boundary='symm')
    by = convolve2d(W_pix*grad_By, flipped_Iy, mode='valid', boundary='symm')
    bt = bx + by

    from skimage.transform import resize
    b = resize(bt, (kh,kw), order=1, preserve_range=True, anti_aliasing=False).ravel()

    K0 = np.ones(kernel_shape)/ (kh*kw)
    Kf = custom_cg(op, b, K0.ravel(), tol=1e-3, maxiter=80)
    K = Kf.reshape(kernel_shape)
    K[K<0]=0
    s=K.sum()
    return K/s if s>1e-8 else np.ones(kernel_shape)/(kh*kw)


def optimize(B, K_init, lam=0.008, beta=2, tmax=50, jmax=4):
    I = B.copy()
    K = K_init.copy()
    for j in range(jmax):
        for t in range(tmax):
            M = compute_latent_map(I, K)
            I = update_I(I, B, K, M, lam)
        M = compute_latent_map(I, K)
        K = update_K(B, I, K, M, beta, kernel_shape=K.shape)
    return I, K


def pyramid_optimize(B, K_init, levels=3, lam=0.008, beta=2, tmax=50, jmax=4):
    pyr, _ = build_pyramid(B, num_levels=levels)
    pyr = pyr[::-1]
    K = K_init.copy()
    from skimage.transform import resize
    for i, B_lv in enumerate(pyr):
        K = K/ K.sum()
        I_lv, K = optimize(B_lv, K, lam, beta, tmax, jmax)
        if i+1<len(pyr):
            K_up = upsample_kernel(K, scale_factor=2)
            K = resize(K_up, K_init.shape, order=1, preserve_range=True, anti_aliasing=False)
            K = K/ K.sum()
    I_final, K_final = optimize(B, K, lam, beta, tmax, jmax)
    return I_final, K_final
