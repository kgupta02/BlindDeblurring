# main_deblurring.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from io_utils import read_image, save_image, save_kernel
from optimization import pyramid_optimize
from nonblind_deconv import nonblind_deconv
from metrics import compute_psnr, compute_ssim
from skimage import io

def main():
    parser = argparse.ArgumentParser(description='Blind image deblurring')
    parser.add_argument('--input', type=str, required=True, help='Path to input (blurred) image')
    parser.add_argument('--output', type=str, default='/results/deblurred.png', help='Path to save output joint image')
    parser.add_argument('--kernel_output', type=str, default='/results/kernel.png', help='Path to save estimated blur kernel')
    parser.add_argument('--lambda_val', type=float, default=0.001, help='Regularization weight for latent image')
    parser.add_argument('--beta_val', type=float, default=2.0, help='Regularization weight for kernel estimation')
    parser.add_argument('--num_levels', type=int, default=5, help='Number of pyramid levels')
    args = parser.parse_args()

    image_files = [fname for fname in os.listdir('../data/')
                 if fname.endswith(('.jpg','.png'))]
    SSIMs = []
    PSNRs = []

    for name in image_files:

        # 1. Read input blurred image
        B = read_image('../data/'+name)

        # 2. Initialize kernel
        kernel_shape = (13, 13)
        K_init = np.ones(kernel_shape, dtype=np.float32) / (kernel_shape[0] * kernel_shape[1])

        # 3. Coarse-to-fine blind deblurring optimization
        I_est, K_est = pyramid_optimize(B, K_init, levels=5, lam=args.lambda_val, beta=args.beta_val)

        # 4. Optional non-blind deconvolution refinement
        I_refined = nonblind_deconv(B, K_est, lambda_val=args.lambda_val)

        # 5. Compute metrics against input for reference
        psnr_val = compute_psnr(B, I_refined)
        ssim_val = compute_ssim(B, I_refined)
        print(f"PSNR between input and deblurred: {psnr_val:.2f} dB")
        print(f"SSIM between input and deblurred: {ssim_val:.4f}")
        PSNRs.append(psnr_val)
        SSIMs.append(ssim_val)
        B_image = io.imread('../data/'+name)
        I_r = nonblind_deconv(B_image[:, :, 0]/ 255.0, K_est, lambda_val=args.lambda_val)
        I_g = nonblind_deconv(B_image[:, :, 1]/ 255.0, K_est, lambda_val=args.lambda_val)
        I_b = nonblind_deconv(B_image[:, :, 2]/ 255.0, K_est, lambda_val=args.lambda_val)
        I_rgb_restored = np.stack([I_r, I_g, I_b], axis=-1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(B_image / 255.0)
        axes[0].axis('off')
        axes[1].imshow(I_rgb_restored)
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig('../results/RGB_'+name,bbox_inches='tight')

        # 6. Create joint image: original | deblurred
        joint = np.concatenate((B, I_refined), axis=1)

        # 7. Save outputs
        save_image('../results/'+name, joint)
        save_kernel('../results/kernel_'+name, K_est)
        
    print(f"mean PSNR is {np.mean(PSNRs)}")
    print(f"mean SSIM is {np.mean(SSIMs)}")

if __name__ == "__main__":
    main()