# main_deblurring.py

import argparse
import numpy as np
from io_utils import read_image, save_image, save_kernel
from optimization import optimize
from nonblind_deconv import nonblind_deconv
from metrics import compute_psnr, compute_ssim

def main():
    parser = argparse.ArgumentParser(description='Blind image deblurring')
    parser.add_argument('--input', type=str, required=True, help='Path to input (blurred) image')
    parser.add_argument('--output', type=str, default='results/deblurred.png', help='Path to save output joint image')
    parser.add_argument('--kernel_output', type=str, default='results/kernel.png', help='Path to save estimated blur kernel')
    parser.add_argument('--lambda_val', type=float, default=0.008, help='Regularization weight for latent image')
    parser.add_argument('--beta_val', type=float, default=2.0, help='Regularization weight for kernel estimation')
    parser.add_argument('--num_levels', type=int, default=5, help='Number of pyramid levels (unused in single-scale)')
    args = parser.parse_args()

    # 1. Read input blurred image
    B = read_image(args.input)

    # 2. Initialize kernel
    kernel_shape = (15, 15)
    K_init = np.ones(kernel_shape, dtype=np.float32) / (kernel_shape[0] * kernel_shape[1])

    # 3. Blind deblurring optimization
    I_est, K_est = optimize(B, K_init, lam=args.lambda_val, beta=args.beta_val)

    # 4. Optional non-blind deconvolution refinement
    I_refined = nonblind_deconv(B, K_est, lambda_val=args.lambda_val)

    # 5. Compute metrics against input for reference
    psnr_val = compute_psnr(B, I_refined)
    ssim_val = compute_ssim(B, I_refined)
    print(f"PSNR between input and deblurred: {psnr_val:.2f} dB")
    print(f"SSIM between input and deblurred: {ssim_val:.4f}")

    # 6. Create joint image: original | deblurred
    joint = np.concatenate((B, I_refined), axis=1)

    # 7. Save outputs
    save_image(args.output, joint)
    save_kernel(args.kernel_output, K_est)

if __name__ == "__main__":
    main()