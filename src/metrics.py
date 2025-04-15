# metrics.py

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def compute_psnr(ref_img, test_img):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images
    """
    mse = np.mean((ref_img - test_img) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value


def compute_ssim(ref_img, test_img):
    """
    Compute the Structural Similarity Index (SSIM) between two images
    """
    if ref_img.ndim == 3 and ref_img.shape[2] == 3:
        ref_gray = cv2.cvtColor((ref_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
        test_gray = cv2.cvtColor((test_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
    else:
        ref_gray = ref_img
        test_gray = test_img
    
    ssim_value, _ = ssim(ref_gray, test_gray, full=True, data_range=ref_gray.max() - ref_gray.min())
    return ssim_value


def compute_error_ratio(ref_img, test_img):
    """
    Computes a relative error ratio between the reference and test image.
    """
    norm_diff = np.linalg.norm(ref_img - test_img)
    norm_ref = np.linalg.norm(ref_img)
    if norm_ref == 0:
        raise ValueError("Cannot compute relative error.")
    error_ratio = (norm_diff / norm_ref) * 100.0 
    return error_ratio


def compute_kernel_similarity(estimated_kernel, ground_truth_kernel):
    """
    Computes a similarity measure between two blur kernels
    """
    est = estimated_kernel.flatten()
    gt = ground_truth_kernel.flatten()
    eps = 1e-8
    cosine_sim = np.dot(est, gt) / (np.linalg.norm(est) * np.linalg.norm(gt) + eps)
    return cosine_sim